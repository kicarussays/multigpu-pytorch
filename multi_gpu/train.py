import os
import time
import datetime
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from model import pyramidnet
import argparse
from tensorboardX import SummaryWriter


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed


torch.set_num_threads(16)

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


def main_multigpu():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main(gpu, ngpus_per_node, args):
    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='../data', train=True, download=True, 
                            transform=transforms_train)

    # train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
    #                           shuffle=True, num_workers=args.num_workers)
    
    """
        For multi-gpu
    """
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    args.gpu = gpu
    device = f'cuda:{gpu}'
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              sampler=train_sampler)
    

    print('==> Making model..')
    torch.cuda.set_device(args.gpu)
    net = pyramidnet()
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
                          momentum=0.9, weight_decay=1e-4)
    
    train(net, criterion, optimizer, train_loader, device)
            

def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    max_epoch = 10
    for epoch in range(max_epoch):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start = time.time()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100 * correct / total
            
            batches_done = epoch * len(train_loader) + batch_idx + 1
            batches_left = max_epoch * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - epoch_start))
            epoch_start = time.time()
            
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] [ACC: %f}] ETA: %s"
                % (epoch+1, max_epoch, batch_idx+1, len(train_loader), train_loss/(batch_idx+1), acc, time_left)
            )
    
    

if __name__=='__main__':
    main_multigpu()