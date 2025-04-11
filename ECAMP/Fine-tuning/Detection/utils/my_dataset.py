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
import ipdb
from tqdm import tqdm
    

class RSNADetectionDataset(Dataset):
    def __init__(self, root, data_volume, split, img_size, transform, max_objects=10):
        super(RSNADetectionDataset, self)

        self.split = split
        self.root = root
        self.img_size = img_size
        self.data_volume = data_volume
        self.max_objects = max_objects
        self.transform = transform

        if self.split == "train":
            if self.data_volume == "1":
                train_data_path = "./RSNA/train_list_1.csv"
            elif self.data_volume == "10":
                train_data_path = "./RSNA/train_list_10.csv"
            else:
                train_data_path = "./RSNA/train_list.csv"
            data_label_csv = train_data_path
        elif self.split == "val":
            val_data_path = "./RSNA/val_list.csv"
            data_label_csv = val_data_path
        else:
            test_data_path = "./RSNA/val_list.csv"
            data_label_csv = test_data_path
        print("data_label_csv: ", data_label_csv)
        
        self.df = pd.read_csv(data_label_csv)
        img_path_list = self.df["image_path"].tolist()
        bbox_list = self.df["bbox"].tolist()
        
        self.img_paths = []
        self.bboxes = []
        for i in range(len(img_path_list)):
            bbox = bbox_list[i]
            bbox = ast.literal_eval(bbox)
            bbox = np.array(bbox)
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            n = new_bbox.shape[0]
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])
            pad = np.zeros((self.max_objects - n, 5))
            new_bbox = np.vstack([new_bbox, pad])
            self.bboxes.append(new_bbox)
            self.img_paths.append(img_path_list[i])

        self.img_paths = np.array(self.img_paths)
        self.bboxes = np.array(self.bboxes)


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        file_name = self.img_paths[index]
        img_path = os.path.join(self.root, file_name)

        x = Image.open(img_path).convert("RGB")
        x = cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB)
        h, w, _ = x.shape
        x = cv2.resize(x, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = Image.fromarray(x, mode="RGB")

        # x = Image.open(img_path).convert("RGB")
        # x = cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2GRAY)
        # h, w = x.shape
        # x = cv2.resize(x, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # x = Image.fromarray(x, mode="L")

        if self.transform is not None:
            x = self.transform(x)
        
        y = self.bboxes[index]
        y[:, 1] /= w
        y[:, 2] /= h
        y[:, 3] /= w
        y[:, 4] /= h

        sample = {"imgs": x, "labels": y}

        return sample



class ObjectCXRDetectionDataset(Dataset):
    def __init__(self, root, data_volume, split, img_size, transform, max_objects=20) -> None:
        super().__init__()

        self.split = split
        self.root = root
        self.img_size = img_size
        self.data_volume = data_volume
        self.max_objects = max_objects
        self.transform = transform

        if self.split == "train":
            if self.data_volume == "1":
                train_data_path = "./ObjectCXR/train_list_1.csv"
            elif self.data_volume == "10":
                train_data_path = "./ObjectCXR/train_list_10.csv"
            else:
                train_data_path = "./ObjectCXR/train_list.csv"
            data_label_csv = train_data_path
        elif self.split == "val":
            val_data_path = "./ObjectCXR/test_list.csv"
            data_label_csv = val_data_path
        else:
            test_data_path = "./ObjectCXR/test_list.csv"
            data_label_csv = test_data_path
        print("data_label_csv: ", data_label_csv)

        self.df = pd.read_csv(data_label_csv)
        self.df = self.df.sort_values(by="image_name")
        img_path_list = self.df["image_name"].tolist()
        bbox_list = self.df["bboxes"].tolist()

        self.filenames_list = []
        self.bboxs_list = []
        for i in range((len(bbox_list))):
            image_path = img_path_list[i]
            # bbox = bbox_list[i]

            # bbox = ast.literal_eval(bbox)
            # bbox = np.array(bbox)

            # padding = np.zeros((max_objects - len(bbox), 4))
            # bbox = np.vstack((bbox, padding))
            # bbox = np.hstack((np.zeros((max_objects, 1)), bbox))

            # new_bbox = bbox.copy()
            # # xminyminwh -> xywh
            # new_bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
            # new_bbox[:, 2] = bbox[:, 2] + bbox[:, 4] / 2

            self.filenames_list.append(image_path)
            # self.bboxs_list.append(new_bbox)
        
        if self.split != "test":
            self.dir = "train"
        else:
            self.dir = "dev"

        # self.images_list = []
        # for item in tqdm(self.filenames_list):
        #     img_dir_path = os.path.join(self.root, self.dir)
        #     img_path = os.path.join(img_dir_path, item)
        #     x = Image.open(img_path).convert("RGB")
        #     x = np.array(x)
        #     self.images_list.append(x)


    def __len__(self):
        return len(self.filenames_list)
    

    def __getitem__(self, index):
        file_name = self.filenames_list[index]
        # x = self.images_list[index]
        # if self.split != "test":
        #     self.dir = "train"
        # else:
        #     self.dir = "dev"
        self.dir = 'resize'
        img_dir_path = os.path.join(self.root, self.dir)
        img_path = os.path.join(img_dir_path, file_name)
        x = Image.open(img_path).convert("RGB")
        x = np.array(x)
        
        h, w, _ = x.shape
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # x = cv2.resize(x, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # import ipdb; ipdb.set_trace()
        # cv2.write("test.jpg", x)
        x = Image.fromarray(x, mode="RGB")
        y = np.load(os.path.join(img_dir_path, file_name+".npy"))

        if self.transform is not None:
            x = self.transform(x)
        
        # y = self.bboxs_list[index]
        # y[:, 1] /= w
        # y[:, 2] /= h
        # y[:, 3] /= w
        # y[:, 4] /= h

        sample = {"imgs": x, "labels": y}

        return sample