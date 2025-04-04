import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dist_util import get_world_size
from .my_dataset import XRAY

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    args.initial_size = 1000

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4722], std=[0.3028])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(int(args.img_size / args.ratio)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4722], std=[0.3028])
    ])


    if args.stage == "test":
        testset = XRAY(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="test", transform= transform_test)
        print("testset",len(testset))
        if args.local_rank == 0:
            torch.distributed.barrier()
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.eval_batch_size//get_world_size(),
                            num_workers=16,
                            pin_memory=True) if testset is not None else None

        return test_loader

    trainset = XRAY(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="train", transform= transform_train)

    valset = XRAY(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="val", transform= transform_test)
    print("train_loader",len(trainset ))
    print("test_loader",len(valset))
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size//get_world_size(),
                              num_workers=16,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size//get_world_size(),
                             num_workers=16,
                             pin_memory=True) if valset is not None else None

    return train_loader, val_loader
