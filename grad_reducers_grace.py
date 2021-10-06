import torch

class SignSGDCompressor(object):

    def __init__(self):
        pass

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        tensor_compressed = tensor >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        sign_encode, = tensors
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        sign = agged_tensor >= 0
        agged_tensor = sign * 2.0 - 1.0
        return agged_tensor

class QSGDCompressor(object):

    def __init__(self, quantum_num):
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(object):

    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)


def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return indices, values


class RandomKCompressor(object):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio):
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)

class TernGradCompressor(object):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape) 


