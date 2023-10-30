import torch
import cv2
import numpy as np
import csv
import torchvision.transforms as transforms
from Data.Dataset_one_by_one import Dataset
from torch.utils import data
import pickle
import torch.nn.functional as F
import os
from agentDetection import detect_Agent,cropping_Agents
from featureExtraction import extract_Features
from gluon_feature_extractor import Feat_Extractor
import json
MXNET_CUDNN_AUTOTUNE_DEFAULT = 0
class AutoDict(dict):
    def __missing__(self, k):
        self[k] = AutoDict()
        return self[k]

Dataset_name = 'ActivityNet'

data_path = "../Datasets/"+Dataset_name+"/rgb-images/" 



params = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
train_list = []
train_label = []
test_list = []
test_label = []
if Dataset_name == 'ROAD':
    img_x, img_y = 960, 1280

if Dataset_name == 'Thumos':
    img_x, img_y = 180*3, 320*3

if Dataset_name == 'ActivityNet':
    img_x, img_y = 240*2, 320*2


transform = transforms.Compose([transforms.Resize([img_x, img_y]),transforms.ToTensor()])

seq_len = 30
dataset = Dataset(Dataset_name,data_path, seq_len,(img_y,img_x), transform=transform)

dataset_loader = data.DataLoader(dataset, **params)

device = torch.device("cuda")
out_feat = AutoDict()

feat_ext = Feat_Extractor()
videos = set()
videos_list = []
snippet_no=0
run_for = 10
for batch_idx, (X,X_list,X_org,video_name,start_frame,end_frame) in enumerate(dataset_loader):
    video_name= video_name[0]
    videos_list.append(video_name)
    if video_name in videos:
        snippet_no +=1
    else:
        if out_feat['database'] != {}:
            vv_name = videos_list[batch_idx-1]
            with open('../Datasets/'+Dataset_name+'/vid_feats_'+str(seq_len)+'/'+Dataset_name+'_features_seq_len_'+str(seq_len)+'_'+vv_name+'_.pkl', 'wb') as output:
                pickle.dump(out_feat, output)            
            out_feat = AutoDict()

        snippet_no = 0
        videos.add(video_name)

    # print(video_name,snippet_no)
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    images_list = []
    [images_list.append(torch.squeeze(s, 0).detach().numpy()) for s in X_list] 
    
    # print(snippet_feats.shape)
    X = X.cuda().to(device)
    agent_tubes = detect_Agent(X,X_org,Dataset_name)
    # print(agent_tubes)
    croped_agents = cropping_Agents(agent_tubes[0],images_list)
    snippet_feats = feat_ext.process(images_list)
    out_feat['database'][video_name]['snippets'][snippet_no]['start_frame'] = int(start_frame)
    out_feat['database'][video_name]['snippets'][snippet_no]['end_frame'] = int(end_frame)
    out_feat['database'][video_name]['snippets'][snippet_no]['seq_len'] = int(seq_len)
    out_feat['database'][video_name]['snippets'][snippet_no]['scene_feat'] = snippet_feats

    for agent in croped_agents:
        out_feat['database'][video_name]['snippets'][snippet_no]['agents'][int(agent)]['class_label'] = croped_agents[agent]['class_label']
        out_feat['database'][video_name]['snippets'][snippet_no]['agents'][int(agent)]['class_index'] = int(croped_agents[agent]['class_index'])
        # print(len(croped_agents[agent]['cropped_frames']))
        # print(croped_agents[agent]['cropped_frames'][0].shape)
        agent_feat = feat_ext.process(croped_agents[agent]['cropped_frames'])
        out_feat['database'][video_name]['snippets'][snippet_no]['agents'][int(agent)]['agent_feat'] = agent_feat
    # print(rr)
    # print(agent_tubes[0])
    print(batch_idx, " out of ", len(dataset_loader))
    # if batch_idx > run_for:
    #     break   

# with open('../Datasets/'+Dataset_name+'/'+Dataset_name+'_features_seq_len_'+str(seq_len)+'_features.json', 'w') as outfile:
#         json.dump(out_feat, outfile)
# with open('../Datasets/'+Dataset_name+'/'+vid_feats+'/'+Dataset_name+'_features_seq_len_'+str(seq_len)+'_'+video_name+'_.pkl', 'wb') as output:
#     pickle.dump(out_feat, output)
