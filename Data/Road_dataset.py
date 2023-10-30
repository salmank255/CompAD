import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

class Dataset_ROAD(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels,size, transform=None):
        "Initialization"
        self.seq_len = 12
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.ids = list()
        self.video_list = list()
        self._make_lists(folders,labels)
        self.size = size


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self,folders,labels):
        for i in range(len(folders)):
            videoname = folders[i]
            label = labels[i]
            self.video_list.append(videoname)
            list = os.listdir(os.path.join(self.data_path, videoname))
            number_files = len(list)
            for frame_num in range(1,number_files-(self.seq_len),self.seq_len):
                self.ids.append([videoname,label, frame_num])

    def __getitem__(self, index):
        #print(index)
        "Generates one sample of data"
        # Select sample
        video,label,start_frame = self.ids[index]
        # print(video,start_frame)
        X = []
        X_org = []
        X_org_ = []
        
        for i in range(start_frame,start_frame+self.seq_len):
            img = cv2.imread(os.path.join(self.data_path, video, 'image_{:05d}.jpg'.format(i)))
            image = img.copy()
            # image = img.copy().transpose((2, 0, 1))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.ascontiguousarray(image)
            if self.transform is not None:
                image = self.transform(Image.fromarray(image))
            X.append(image)
            X_org.append(torch.tensor(cv2.resize(img,self.size)))

        X = torch.stack(X, dim=0)
        X_org = torch.stack(X_org, dim=0)
        y = torch.LongTensor([int(label)])
        return X, y,X_org,video

    def custum_collate(batch):
        images = []
        images_org = []
        labels = []
        video_name = []
        for sample in batch:
            images.append(sample[0])
            labels.append(sample[1])
            images_org.append(sample[2])
            video_name.append(sample[3])
            
        return images, labels,images_org,video_name