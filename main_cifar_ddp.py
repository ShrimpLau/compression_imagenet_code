
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

import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD
import gradient_reducers
import s3_utils
from timer import Timer

def metric(*args, **kwargs):
    if True == 0:
        log_metric(*args, **kwargs)

import cifar_architectures
timer = Timer(verbosity_level=2, log_fn=metric)

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

def _get_compression_param(reducer_name, device, reducer_param):
    if reducer_name == "PowerSGD":
        reducer = gradient_reducers.RankKReducer(random_seed=42,
                                                  device=device,
                                                  timer=timer,
                                                  n_power_iterations=0,
                                                  reuse_query=True,
                                                  rank = reducer_param)
    if reducer_name == "SignSGD":
        reducer = gradient_reducers.SignSGDwithMajorityVoteReducer(random_seed=42,
                                                 device=device,
                                                 timer=timer)
    if reducer_name == "Topk":
        reducer = gradient_reducers.GlobalTopKReducer(random_seed=42,
                                                      device=device,
                                                      timer=timer,
                                                      compression=reducer_param)

    if reducer_name == "MSTopK":
        reducer = gradient_reducers.MsTopKReducer(random_seed=42,
                                                  device=device, timer=timer,
                                                  k=reducer_param)

    return reducer

def powersgd_single_call(args, psgd_rank, bsize, network_name):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = getattr(cifar_architectures, network_name)()
    # model = models.__dict__[network_name]()
    model.to(assigned_device)

    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    state = PowerSGD.PowerSGDState(process_group=None,
                                    matrix_approximation_rank=psgd_rank,
                                    start_powerSGD_iter=3)
    
    model.register_comm_hook(state, PowerSGD.powerSGD_hook) 
    
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    data = torch.randn((bsize, 3, 32, 32))
    target = torch.randint(0,9, [bsize])

    for batch_idx in range(100):
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize()
        start_time.record() 
        loss.backward() #we have the gradients
        # grad_list = [p.grad for p in model.parameters()]
        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        # reducer.reduce(send_buffers, grad_list, memories)
        # we have the gradients synchronized
        stop_time.record() 
        torch.cuda.synchronize()
        # print ("Time {}, Device {}".format(start_time.elapsed_time(stop_time),
                                         # args.device))
        time_list.append(start_time.elapsed_time(stop_time))
        if batch_idx == 30:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "{}_cifar10_powersgd_overlap_rank_{}_out_file_{}_batch_size_{}.json".format(network_name, psgd_rank,
                                                                                          global_rank,
                                                                                          bsize)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name,
                                    "{}/{}".format(args.s3_prefix, file_name))

            print ("Done {}".format(network_name))
            break

def powersgd_serial_originial(args, psgd_rank, bsize, network_name):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = getattr(cifar_architectures, "ResNet18")()
    
    model.to(assigned_device)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]
     
    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    reducer = _get_compression_param("PowerSGD", assigned_device, psgd_rank)
    
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    data = torch.randn((bsize, 3, 224, 224))
    target = torch.randint(0,9, [bsize])

    for batch_idx in range(100):
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize()
        start_time.record() 
        loss.backward() #we have the gradients
            for grad, memory, send_bfr in zip([p.grad for p in
                                               model.parameters()], memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)
        # we have the gradients synchronized
        stop_time.record() 
        torch.cuda.synchronize()
        # print ("Time {}, Device {}".format(start_time.elapsed_time(stop_time),
                                         # args.device))
        time_list.append(start_time.elapsed_time(stop_time))
        model.zero_grad()
        if batch_idx == 30:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "{}_cifar10_powersgd_serial_rank_{}_out_file_{}_batch_size_{}.json".format(network_name, psgd_rank,
                                                                                           global_rank,
                                                                                           bsize)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name,
                                    "{}/{}".format(args.s3_prefix, file_name))

            print ("Done Resnet 101")
            break

def ddp_training(args, bsize, network_name):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    model = getattr(cifar_architectures, network_name)()
    # model = models.__dict__[network_name]()
    model.to(assigned_device)


    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    # state = PowerSGD.PowerSGDState(process_group=None,
                                    # matrix_approximation_rank=psgd_rank,
                                    # start_powerSGD_iter=3)
    
    # model.register_comm_hook(state, PowerSGD.powerSGD_hook) 
    
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    data = torch.randn((bsize, 3, 32, 32))
    target = torch.randint(0,9, [bsize])

    for batch_idx in range(100):
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize()
        start_time.record() 
        loss.backward() #we have the gradients
        # grad_list = [p.grad for p in model.parameters()]
        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        # reducer.reduce(send_buffers, grad_list, memories)
        # we have the gradients synchronized
        stop_time.record() 
        torch.cuda.synchronize()
        # print ("Time {}, Device {}".format(start_time.elapsed_time(stop_time),
                                         # args.device))
        time_list.append(start_time.elapsed_time(stop_time))
        if batch_idx == 30:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "{}_ddp_cifar10_out_file_{}_batch_size_{}.json".format(network_name,
                                                                                          global_rank,
                                                                                          bsize)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name,
                                    "{}/{}".format(args.s3_prefix, file_name))

            print ("Done {}".format(network_name))
            break

if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    log_file_name = os.path.basename(args.log_file).split(".")[0]+"_args_logged_{}.log".format(9)
    # timing_logging = os.path.basename(args.log_file).split(".")[0]+"_time_logged_{}.json".format(args.device)
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Arguments: {}".format(args))
    print ("In If")
    print (args)
    dist.init_process_group(backend="NCCL", init_method="env://")
    print ("Dist connected")
    powersgd_single_call(args, 4, 128, "ResNet18")
    powersgd_single_call(args, 8, 128, "ResNet18")
    powersgd_single_call(args, 16, 128, "ResNet18")

    print("powersgd single call") 
    powersgd_serial_originial(args, 4, 128, "ResNet18")
    powersgd_serial_originial(args, 8, 128, "ResNet18")
    powersgd_serial_originial(args, 16, 128, "ResNet18")
    print ("powersgd serial")
    ddp_training(args, 128, "ResNet18")
    print ("powersgd ddp")
