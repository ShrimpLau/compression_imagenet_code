import os
import sys
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
import s3_utils
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
    # parser.add_argument("--arch", default="resnet50", type=str,
                        # help="network type")
    # parser.add_argument("--master-ip", type=str, help="Ip address of master")
    parser.add_argument("--local_rank", type=int, help="Rank of the experiment")
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--dataset-location", type=str, help="Data path")
    parser.add_argument("--loader-threads", type=int, default=2, help="Loader threads")
    # parser.add_argument("--device", type=str, default="cuda:0", 
                        # help="GPU to use")
    parser.add_argument("--log-file", type=str, default="Log file")
    parser.add_argument("--num-workers", type=int, 
                        help="Number of total  workers")
    parser.add_argument("--s3-prefix", type=str, default=None, 
                        help="s3-prefix to write")
    parser.add_argument("--node_rank", type=int)
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

def _get_compression_param(reducer_name, device, reducer_param):
    if reducer_name == "PowerSGD":
        reducer = gradient_reducers.RankKReducer(random_seed=42,
                                                  device=device,
                                                  timer=timer,
                                                  n_power_iterations=0,
                                                  reuse_query=True,
                                                  rank = reducer_param)
    return reducer


def main_resnet50(args):
    #Initialize dataset
    
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = models.__dict__["resnet50"]()
    model.to(assigned_device)

    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    # train_loader = _create_data_loader(args)
    #reducer = _get_compression_param(args)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()
    # for batch_idx, (data, target) in enumerate(train_loader):
    data = torch.randn((args.batch_size, 3, 224, 224))
    target = torch.randint(0,900, [args.batch_size])
    for batch_idx in range(100):
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize() #let's sync before starting
        start_time.record() 
        loss.backward() #we have the gradients
        stop_time.record() 
        torch.cuda.synchronize()
        time_list.append(start_time.elapsed_time(stop_time))
        if batch_idx == 30:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "resnet50_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name,
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Res 50 done")
            break

            # sys.exit(0)


def main_resnet101(args):
    #Initialize dataset
    
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = models.__dict__["resnet101"]()
    model.to(assigned_device)

     
    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    # train_loader = _create_data_loader(args)
    # reducer = _get_compression_param(args)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()
    # for batch_idx, (data, target) in enumerate(train_loader):
    data = torch.randn((args.batch_size, 3, 224, 224))
    target = torch.randint(0,900, [args.batch_size])
    for batch_idx in range(100):
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize() #let's sync before starting
        start_time.record() 
        loss.backward() #we have the gradients
        stop_time.record() 
        torch.cuda.synchronize()
        time_list.append(start_time.elapsed_time(stop_time))
        if batch_idx == 30:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "resnet101_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name,
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done res 101")
            break
            # sys.exit(0)

def powersgd_resnet50(args, psgd_rank):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = models.__dict__["resnet101"]()
    model.to(assigned_device)

     
    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    reducer = _get_compression_param("PowerSGD", assigned_device, psgd_rank)
    
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    data = torch.randn((args.batch_size, 3, 224, 224))
    target = torch.randint(0,900, [args.batch_size])

if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    # log_file_name = os.path.basename(args.log_file).split(".")[0]+"_args_logged_{}.log".format(args.device)
    # timing_logging = os.path.basename(args.log_file).split(".")[0]+"_time_logged_{}.json".format(args.device)
    # logging.basicConfig(filename=log_file_name)
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # logger.info("Arguments: {}".format(args))
    print ("In If")
    print (args)
    dist.init_process_group(backend="NCCL", init_method="env://")
    print ("Dist connected")
    main_resnet50(args)
    main_resnet101(args)

