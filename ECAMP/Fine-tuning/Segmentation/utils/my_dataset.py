import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import CenterCrop
import numpy as np 
import pandas as pd
import ast
import random
from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2


def resize_img(img, scale: int):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


class SIIMSegmentDataset(Dataset):
    
    def __init__(self, root, data_volume, split, img_size, num_channels):
        super(SIIMSegmentDataset, self)

        self.split = split
        self.root = root
        self.img_size = img_size
        self.num_channels = num_channels
        self.data_volume = data_volume
        self.transform = self.get_transforms()
        
        if self.split == "train":
            if self.data_volume == "1":
                train_data_path = "./datasets/SIIM/train_list_1.csv"
            elif self.data_volume == "10":
                train_data_path = "./datasets/SIIM/train_list_10.csv"
            else:
                train_data_path = "./datasets/SIIM/train_list.csv"
            data_label_csv = train_data_path
        
        elif self.split == "val":
            val_data_path = "./datasets/SIIM/val_list.csv"
            data_label_csv = val_data_path
                 
        else:
            test_data_path = "./datasets/SIIM/test_list.csv"
            data_label_csv = test_data_path
           
        self.df = pd.read_csv(data_label_csv)


        self.df["ImagePath"] = self.df["ImageId"].apply(
            lambda x: os.path.join(self.root, "train/images/1024/dicom/" + x + ".png")
        )
        
        # only keep positive samples for segmentation
        self.df["class"] = self.df["EncodedPixels"].apply(lambda x: x != "-1")
        if self.split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos = self.df_pos["ImageId"].nunique()
            neg_series = self.df_neg["ImageId"].unique()
            neg_series_selected = np.random.choice(
                neg_series, size=n_pos, replace=False
            )
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])
        
        self.img_ids = self.df["ImageId"].unique().tolist()



    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_id_df = self.df.groupby("ImageId").get_group(img_id)

        # get image
        img_path = img_id_df.iloc[0]["ImagePath"]
        if self.num_channels == 1:
            x = Image.open(img_path).convert("L")
        else:
            x = Image.open(img_path).convert("RGB")
        x = np.array(x)

        # get labels
        rle_list = img_id_df["EncodedPixels"].tolist()
        mask = np.zeros([1024, 1024])
        if rle_list[0] != "-1":
            for rle in rle_list:
                mask += self.rle2mask(rle, 1024, 1024)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.img_size)

        augmented = self.transform(image=x)
        x = augmented["image"]
        y = transforms.ToTensor()(mask).squeeze()

        return x, y


    def __len__(self):
        
        return len(self.img_ids)
    

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.array([int(x) for x in rle.split(" ")])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position: current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T
    

    def get_transforms(self, mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028)):
        list_transforms = []
        if self.num_channels == 1:
            mean = (0.4722,)
            std = (0.3028,)
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )


        img_transforms = Compose(list_transforms)
        return img_transforms
    

class RSNASegmentDataset(Dataset):
    def __init__(self, root, data_volume, split, img_size, num_channels):
        super(RSNASegmentDataset, self)

        self.split = split
        self.root = root
        self.img_size = img_size
        self.data_volume = data_volume
        self.num_channels = num_channels

        if self.split == "train":
            if self.data_volume == "1":
                train_data_path = "./datasets/RSNA/train_list_1.csv"
            elif self.data_volume == "10":
                train_data_path = "./datasets/RSNA/train_list_10.csv"
            else:
                train_data_path = "./datasets/RSNA/train_list.csv"
            data_label_csv = train_data_path
        elif self.split == "val":
            val_data_path = "./datasets/RSNA/val_list.csv"
            data_label_csv = val_data_path
        else:
            test_data_path = "./datasets/RSNA/test_list.csv"
            data_label_csv = test_data_path
        
        self.df = pd.read_csv(data_label_csv)
        self.img_paths = self.df["image_path"].tolist()
        self.bboxes = self.df["bbox"].tolist()

        self.transform = self.get_transforms()


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        file_name = self.img_paths[index]
        img_path = os.path.join(self.root, file_name)
        if self.num_channels == 1:
            x = Image.open(img_path).convert("L")
        else:
            x = Image.open(img_path).convert("RGB")
        x = np.array(x)
        mask = np.zeros((1024, 1024))
        bbox = self.bboxes[index]
        bbox = ast.literal_eval(bbox)
        bbox = np.array(bbox)
        new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)
        if len(new_bbox) > 0:
            for i in range(len(new_bbox)):
                try:
                    mask[new_bbox[i, 1]:new_bbox[i, 3],
                         new_bbox[i, 0]:new_bbox[i, 2]] += 1
                except:
                    import ipdb; ipdb.set_trace()
        mask = (mask >= 1).astype("float32")
        # mask = resize_img(mask, self.img_size)
        augmented = self.transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"].squeeze()

        return x, y



    def get_transforms(self, mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028)):
        list_transforms = []

        if self.num_channels == 1:
            mean = (0.4722,)
            std = (0.3028,)

        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms


class RIGASegmentDataset(Dataset):
    def __init__(self, root, data_volume, img_size, split):
        super(RIGASegmentDataset, self)

        self.split = split
        self.root = root
        self.img_size = img_size
        self.data_volume = data_volume

        if self.split == "train":
            if self.data_volume == "1":
                train_data_path = "./datasets/RIGA/train_list_1.csv"
            elif self.data_volume == "10":
                train_data_path = "./datasets/RIGA/train_list_10.csv"
            else:
                train_data_path = "./datasets/RIGA/train_list.csv"
            data_label_csv = train_data_path
        elif self.split == "val":
            val_data_path = "./datasets/RIGA/val_list.csv"
            data_label_csv = val_data_path
        else:
            test_data_path = "./datasets/RIGA/test_list.csv"
            data_label_csv = test_data_path
        
        self.df = pd.read_csv(data_label_csv)
        self.img_paths = self.df["img_path"].tolist()
        self.mask_paths = self.df["mask_path"].tolist()

        self.train_transform = self.train_transforms()
        self.transform = self.base_transforms()


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_path = os.path.join(self.root, img_name)
        x = Image.open(img_path).convert("RGB")
        x = np.array(x)
        mask_name = self.mask_paths[index]
        mask_path = os.path.join(self.root, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = resize_img(mask, self.img_size)
        if mask.max() > 1:
            mask = mask / 255.0

        disc = mask.copy()
        disc[disc != 0] = 1
        cup = mask.copy()
        cup[cup != 1] = 0
        mask = np.stack((disc, cup))

        if self.split == "train":
            augmented = self.train_transform(image=x, mask=mask)
            augmented_train = self.transform(image=augmented["image"])
            x = augmented_train["image"]
            y = torch.tensor(augmented["mask"])
        else:
            augmented = self.transform(image=x)
            x = augmented["image"]
            y = torch.tensor(mask)


        return x, y



    def train_transforms(self, mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028)):
        list_transforms = []

        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )

        list_trfms = Compose(list_transforms)
        return list_trfms
    

    def base_transforms(self, mean=(0.4722, 0.4722, 0.4722), std=(0.3028, 0.3028, 0.3028)):
        list_transforms = []

        list_transforms.extend(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms


if __name__ == "__main__":
    # data = RIGASegmentDataset("../../../../Data/RIGA/DiscRegion/", "1", 224, "train")
    # print(data[0])

    data = RSNASegmentDataset("../../../../Data/RSNA", "1", "train", 224, 3)
    print(data[0])