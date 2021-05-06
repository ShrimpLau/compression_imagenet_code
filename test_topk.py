import torch


def mstopk(grad_in):
    grad_1d = grad_in.reshape(-1) #reshaping to 1d
    a = torch.abs(grad_1d)
    a_hat = torch.mean(a)
    u = torch.max(a)
    l = 0
    r = 1
    k1 = 0
    k2 = len(grad_1d)
    thres1 = 0
    thres2 = 0
    for i in range(20):
        ratio = l + (r-l)/2
        thres = a_hat + ratio*(u-a_hat)
        nnz = torch.count_nonzero(a >= thres)
        if nnz <= 4:
            r = ratio
            if nnz > k1:
                k1 = nnz
                thres1 = thres
        elif nnz > 4:
            l= ratio
            if nnz < k2:
                k2 = nnz
                thres2 = thres
    l1 = torch.nonzero(a>= thres1, as_tuple=True)[0] #since 1d no problem
    l2 = torch.nonzero((a<thres1) & (a >= thres2), as_tuple=True)[0]
    rand = random.randint(0, len(l2)-(4-k1)+1)
    l = torch.concat(l1, l2[rand:rand+4-k1])
    kai = grad_in[l]
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    mstopk(torch.range(0,10,step=0.1))
