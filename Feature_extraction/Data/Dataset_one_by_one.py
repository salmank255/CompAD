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
import glob

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self,Dataset_name, data_path,seq_len, size, transform=None):
        "Initialization"
        self.Dataset_name = Dataset_name
        self.seq_len = seq_len
        self.data_path = data_path
        self.transform = transform
        self.ids = list()
        self.video_list = list()
        self._make_lists(self.data_path)
        self.size = size

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self,data_path):
        for videoname in os.listdir(data_path):
            if os.path.isfile('../Datasets/'+self.Dataset_name+'/vid_feats_'+str(self.seq_len)+'/'+self.Dataset_name+'_features_seq_len_'+str(self.seq_len)+'_'+videoname+'_.pkl'):
                continue
            self.video_list.append(videoname)
            number_files = len(glob.glob(data_path+'/'+videoname+'/*.jpg'))
            for frame_num in range(1,number_files-(self.seq_len),self.seq_len):
                self.ids.append([videoname,frame_num])

    def __getitem__(self, index):
        #print(index)
        "Generates one sample of data"
        # Select sample
        video,start_frame = self.ids[index]
        # print(video,start_frame)
        end_frame = start_frame+self.seq_len-1
        X = []
        X_org = []
        X_list = []
        
        for i in range(start_frame,start_frame+self.seq_len):
            img = cv2.imread(os.path.join(self.data_path, video, '{:05d}.jpg'.format(i)))
            image = img.copy()
            X_list.append(cv2.resize(img.copy(),self.size))
            # if self.Dataset_name == 'Thumos':
            #     X_list.append(cv2.resize(img.copy(),(224,224)))
            # else:
            #     X_list.append(img.copy())

            # image = img.copy().transpose((2, 0, 1))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.ascontiguousarray(image)
            if self.transform is not None:
                image = self.transform(Image.fromarray(image))
            X.append(image)
            X_org.append(torch.tensor(cv2.resize(img,self.size)))

        X = torch.stack(X, dim=0)
        X_org = torch.stack(X_org, dim=0)
        return X,X_list,X_org,video,start_frame,end_frame

    def custum_collate(batch):
        images = []
        images_list = []
        images_org = []
        video_name = []
        start_frame = []
        end_frame = []
        for sample in batch:
            images.append(sample[0])
            images_list.append(sample[1])
            images_org.append(sample[2])
            video_name.append(sample[3])
            start_frame.append(sample[4])
            end_frame.append(sample[5])
            
        return images,images_list, images_org,video_name,start_frame,end_frame

