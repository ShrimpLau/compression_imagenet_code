import os
import time
import numpy as np
import argparse
import logging
import json
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from collections import defaultdict
import torchvision.models as models
from torch.autograd import Variable

import gradient_reducers
from timer import Timer

def metric(*args, **kwargs):
    if True == 0:
        log_metric(*args, **kwargs)

timer = Timer(verbosity_level=2, log_fn=metric)

imagenet_config = {
    "name": "imagenet",
    "arch": "resnet18",
    "dataset": "imagenet",
    "data_path": "", #TODO
}


def parse_args(parser):
    parser.add_argument("--arch", default="resnet18", type=str,
                        help="network type")
    parser.add_argument("--master-ip", type=str, help="Ip address of master")
    parser.add_argument("--rank", type=int, help="Rank of the experiment")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of total  workers")
    parser.add_argument("--reducer", type=str,
                        help="tell which reducer to use")
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--dataset-location", type=str, help="Data path")
    parser.add_argument("--loader-threads", type=int, help="Loader threads")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="GPU to use")
    parser.add_argument("--log-file", type=str, default="Log file")
    parser.add_argument("--reducer", type=str, default="RankKReduce", 
                        help="Rank to use")
    parser.add_argument("--reducer-param", default=None, 
                        help="extra compression parameter if any")
    args = parser.parse_args()
    return args

def _create_data_loader(args):
    train_dir = os.path.join(args.dataset_location, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.loader_threads,
                                               pin_memory=True,
                                               sampler=sampler)
    return train_loader


def _get_compression_param(args):
    if args.reducer == "PowerSGD":
        reducer = gradient_reducers.RankKReducers(random_seed=42,
                                                  device=args.device,
                                                  timer=timer,
                                                  n_power_iterations=1,
                                                  reuse_query=True,
                                                  rank = args.reducer_param)
    return reducer

def main(args, timing_logging):
    #Initialize dataset 
    dist.init_process_group(backend="NCCL", init_method=args.master_ip, 
                            timeout=datetime.timedelta(seconds=120),
                            world_size=args.num_workers, rank=args.rank)
    model = models.__dict__[args.arch]()
    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    train_loader = _create_data_loader(args)
    reducer = _get_compression_param(args)
    self.model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = data.to(args.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        start_time.record() 
        loss.backward() #we have the gradients
        grad_list = [p.grad for p in model.parameters()]
        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grads, memories)
        # we have the gradients synchronized
        stop_time.record() 
        torch.cuda.synchronize()
        print ("Time {}".format(start_time.elapsed_time(stop_time)))
        if batch_idx == 5:
            sys.exit(0)

    # training done


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    args_logging = os.path.basename(args.log_file).split(".")[0]+"_args_logged.log"
    timing_logging = os.path.basename(args.log_file).split(".")[0]+"_time_logged.json"
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Arguments: {}".format(args))
    print (args)
    main(args, timing_logging)

