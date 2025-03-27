
import os
import torch
import torch.utils.data as data
import numpy as np 
import pandas as pd 

from PIL import Image
import cv2
import ipdb

class XRAY(data.Dataset):
    
    def __init__(self, root, data_volume, task, split="train", transform=None):
        super(XRAY, self)
        if data_volume == '1':
            train_label_data = "train_list_1.txt"
        if data_volume == '10':
            train_label_data = "train_list_10.txt"
        if data_volume == '100':
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.task = task
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
                 
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data
           
        #---- Open file, get image paths and labels
        
        fileDescriptor = open(os.path.join("./datasets/" + self.task, downloaded_data_label_txt), "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                imagePath = os.path.join(root, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        # ipdb.set_trace()
        fileDescriptor.close()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)

        # imageData = self.rgb2gray(imageData).unsqueeze(0)               # APTOS
        # imageData = torch.cat((imageData, imageData, imageData), 0)     # APTOS
         
        return imageData, imageLabel

    def __len__(self):
        
        return len(self.listImagePaths)


    def rgb2gray(self, image):
        image = image[0, :, :]*0.299 + image[1, :, :]*0.587 + image[2, :, :]*0.114

        return image
        
