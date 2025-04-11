import logging

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dist_util import get_world_size
from .my_dataset import RSNADetectionDataset, ObjectCXRDetectionDataset

logger = logging.getLogger(__name__)


class DetectionDataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224, jitter_strength: float = 1.):
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            )

            kernel_size = int(0.1 * 224)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028))
            ]

            # data_transforms = [
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0.4722, ), std=(0.3028, ))
            # ]
        else:
            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028))
            ]
            # data_transforms = [
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0.4722, ), std=(0.3028, ))
            # ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     # type: ignore

    if args.stage == "test":
        testset = None
        if args.task == "RSNA":
            testset = RSNADetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="test",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=False),
            )
        else:
            testset = ObjectCXRDetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="test",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=False),
            )

        print("testset size: ",len(testset)) # type: ignore
        
        if args.local_rank == 0:
            torch.distributed.barrier() # type: ignore
        
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.eval_batch_size//get_world_size(),
                            num_workers=16,
                            pin_memory=True) if testset is not None else None

        return test_loader

    else:
        trainset = None
        valset = None
        if args.task == "RSNA":
            trainset = RSNADetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="train",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=True),
            )
            valset = RSNADetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="val",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=False),
            )
        else:
            trainset = ObjectCXRDetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="train",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=True),
            )
            valset = ObjectCXRDetectionDataset(
                root=args.dataset_path,
                data_volume=args.data_volume,
                split="val",
                img_size=args.img_size,
                transform=DetectionDataTransforms(is_train=False),
            )
        trainset.__getitem__(0)

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
