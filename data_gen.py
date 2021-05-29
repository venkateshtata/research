import h5py
import numpy as np
import tqdm
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image


class drivingDataset(Dataset):

    def __init__(self, transform=None):
        self.log = h5py.File("log/2016-06-08--11-46-01.h5", "r")
        self.cam = h5py.File("cam/2016-06-08--11-46-01.h5", "r")
        self.transform = transform

        # print("max : ", -(max(self.log['steering_angle']))/10.0)
        # print("min : ", -(min(self.log['steering_angle']))/10.0)
        # self.cam_data = []
        # if(datasetType=="train"):
        #     self.cam_data = self.cam['X'][:14526]
        # if(datasetType=="test"):
        #     self.cam_data = self.cam['X'][14526:]
        # self.transform = transform


    def __len__(self):
        return(len(self.cam['X']))

    def __getitem__(self, index):
        img = torch.Tensor(self.cam['X'][int(self.log['cam1_ptr'][index])])
        label = self.log['steering_angle'][index]

        if(self.transform):
            img = self.transform(img)

        return(img, label)
