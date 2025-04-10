import logging

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dist_util import get_world_size
from .my_dataset import SIIMSegmentDataset, RSNASegmentDataset, RIGASegmentDataset

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     # type: ignore

    if args.name == "prior":
        num_channels = 1
    else:
        num_channels = 3


    if args.stage == "test":
        testset = None
        if args.task == "SIIM":
            testset = SIIMSegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="test",
                img_size=args.img_size,
                num_channels=num_channels
            )
        elif args.task == "RSNA":
            testset = RSNASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="test",
                img_size=args.img_size,
                num_channels=num_channels
            )
        elif args.task == "RIGA":
            testset = RIGASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="test",
                img_size=args.img_size
            )

        print("testset size: ",len(testset)) # type: ignore
        
        if args.local_rank == 0:
            torch.distributed.barrier() # type: ignore
        
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.eval_batch_size//get_world_size(),
                            num_workers=16,
                            pin_memory=True,
                            prefetch_factor=2) if testset is not None else None

        return test_loader

    else:
        trainset = None
        valset = None
        if args.task == "SIIM":
            trainset = SIIMSegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="train",
                img_size=args.img_size,
                num_channels=num_channels
            )
            valset = SIIMSegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="val",
                img_size=args.img_size,
                num_channels=num_channels
            )
        elif args.task == "RSNA":
            trainset = RSNASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="train",
                img_size=args.img_size,
                num_channels=num_channels
            )
            valset = RSNASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="val",
                img_size=args.img_size,
                num_channels=num_channels
            )
        elif args.task == "RIGA":
            trainset = RIGASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="train",
                img_size=args.img_size
            )
            valset = RIGASegmentDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="val",
                img_size=args.img_size
            )
    
        print("trainset size: ",len(trainset)) # type: ignore
        print("valset size: ",len(valset)) # type: ignore
    
        if args.local_rank == 0:
            torch.distributed.barrier() # type: ignore

        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset) # type: ignore
        val_sampler = SequentialSampler(valset)
        train_loader = DataLoader(trainset, # type: ignore
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
