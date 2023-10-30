# from __future__ import annotations

import math
import os
import numpy as np
from PIL import Image
from torch.utils import data
import json
import os
import pickle
import torch

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path,dataset_name,seq_len,edges_type,no_of_classes,split='train'):
        "Initialization"
        self.seq_len = seq_len
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.edges_type = edges_type
        self.no_of_classes = no_of_classes
        self.nb_samples = np.zeros(self.no_of_classes,dtype=int)
        self.ids = list()
        self.gt_json_path = os.path.join(self.data_path,self.dataset_name, self.dataset_name+'_gt.json')
        self.feat_json_path = os.path.join(self.data_path,self.dataset_name, self.dataset_name+'_features_seq_len_'+str(self.seq_len)+'_.pkl')
        self._make_lists()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self):
        
        with open(self.gt_json_path,'r') as fff:
            self.gt_json = json.load(fff)
       
        with open(self.feat_json_path,'rb') as ff:
            self.feat_json = pickle.load(ff)

        for video in self.feat_json['database']:
            if video in self.gt_json['database'].keys():
                if self.gt_json['database'][video]['subset'] == self.split:
                    for snippets in self.feat_json['database'][video]:
                        for snippet in self.feat_json['database'][video][snippets]:
                            start_frame = self.feat_json['database'][video][snippets][snippet]['start_frame']
                            end_frame = self.feat_json['database'][video][snippets][snippet]['end_frame']
                            mid_frame = math.floor((start_frame+end_frame)/2)
                            annotations = self.gt_json['database'][video]['annotations']
                            for anno in annotations:
                                # print(mid_frame,anno['segment'][0],anno['segment'][1])
                                if mid_frame >= anno['segment'][0] and mid_frame <= anno['segment'][1]:
                                    act_class = anno['class_#']
                                    
                                    if (mid_frame - anno['segment'][0]) <= self.seq_len:
                                        is_act_start = 1
                                    else:
                                        is_act_start = 0
                                    if (anno['segment'][1] - mid_frame) <= self.seq_len:
                                        is_act_end = 1
                                    else:
                                        is_act_end = 0
                                    break
                                else:
                                    act_class = 0
                                    is_act_start = 0
                                    is_act_end = 0
                            self.nb_samples[act_class] +=1
                            self.ids.append([video,snippet,act_class,is_act_start,is_act_end])
    
    def scene_with_same_label_edges(self,no_of_nodes,node_labels):
        edges = []
        source_node_id = 0
        ###self_edge
        edges.append([source_node_id,source_node_id])
        target_ids = np.arange(1,no_of_nodes)
        for target_node_id in target_ids:
            edges.append([source_node_id,target_node_id]) 

        for k in range(1,len(node_labels)):
            for l in range(1,len(node_labels)):
                if node_labels[k] == node_labels[l]:
                    edges.append([k,l])
        edges = torch.tensor(np.transpose(np.array(edges)))
        return edges

    def scene_edges(self,no_of_nodes,node_labels):
        edges = []
        source_node_id = 0
        ###self_edge
        edges.append([source_node_id,source_node_id])
        target_ids = np.arange(1,no_of_nodes)
        for target_node_id in target_ids:
            edges.append([source_node_id,target_node_id])        
        edges = torch.tensor(np.transpose(np.array(edges)))
        return edges

    # def scene_edges(self,no_of_nodes,node_labels):
    #     edges = []
    #     source_node_id = 0
    #     ###self_edge
    #     edges.append([source_node_id,source_node_id])
    #     target_ids = np.arange(1,no_of_nodes+1)
    #     for target_node_id in target_ids:
    #         edges.append([source_node_id,target_node_id])        
    #     edges = np.array(edges)
    #     return edges

    # def fully_connected_edges(self,no_of_nodes,node_labels):
    #     edges = []
    #     # node_ids = np.arange(0,no_of_nodes+1)
    #     for source_node_id in node_labels:
    #         for target_node_id in node_labels:
    #             edges.append([source_node_id,target_node_id]) 
    #     edges = np.array(edges)
    #     return edges

    def fully_connected_edges(self,no_of_nodes,node_labels):
        edges = []
        node_ids = np.arange(0,no_of_nodes)
        for source_node_id in node_ids:
            for target_node_id in node_ids:
                edges.append([source_node_id,target_node_id]) 
        edges = torch.tensor(np.transpose(np.array(edges)))
        return edges

    def get_node_labels(self,agents):
        node_lab = []
        node_lab.append(0)
        for i in range(len(agents)):
            if len(agents[i]) >0:
                node_lab.append(agents[i]['class_index']+1)
        # print(node_lab)
        # print(len(agents))
        return node_lab

    def get_combine_feat_vector(self,scene_feat,agents):
        combine_feat = []
        combine_feat.append(scene_feat)
        for i in range(len(agents)):
            if len(agents[i]) >0:
                combine_feat.append(agents[i]['agent_feat'])
        
        combine_feat = np.squeeze(np.array(combine_feat))
        if combine_feat.shape[0] == 2048:
            combine_feat = np.expand_dims(combine_feat,axis=0)
        return torch.tensor(combine_feat)

    # def __getitem__(self, index):
    #     "Generates one sample of data"
    #     # Select sample
    #     video,snippet,act_class,is_act_start,is_act_end = self.ids[index]
    #     scene_feat = self.feat_json['database'][video]['snippets'][snippet]['scene_feat']
    #     agents = self.feat_json['database'][video]['snippets'][snippet]['agents']
    #     no_of_nodes = len(agents)+1
    #     node_labels = self.get_node_labels(agents)
    #     if self.edges_type == 'scene':
    #         edges = self.scene_edges(no_of_nodes,node_labels)
    #     elif self.edges_type == 'fully_connected': 
    #         edges = self.fully_connected_edges(no_of_nodes,node_labels)
    #     elif self.edges_type == 'scene_same_label': 
    #         edges = self.scene_with_same_label_edges(no_of_nodes,node_labels)
    #     combine_features = self.get_combine_feat_vector(scene_feat,agents)
    #     return np.squeeze(scene_feat),agents,combine_features,edges,node_labels,no_of_nodes,act_class,is_act_start,is_act_end

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        video,snippet,act_class,is_act_start,is_act_end = self.ids[index]
        scene_feat = self.feat_json['database'][video]['snippets'][snippet]['scene_feat']
        agents = self.feat_json['database'][video]['snippets'][snippet]['agents']
        combine_features = self.get_combine_feat_vector(scene_feat,agents)
        no_of_nodes = combine_features.shape[0]
        node_labels = self.get_node_labels(agents)
        if self.edges_type == 'scene':
            edges = self.scene_edges(no_of_nodes,node_labels)
        elif self.edges_type == 'fully_connected': 
            edges = self.fully_connected_edges(no_of_nodes,node_labels)
        elif self.edges_type == 'scene_same_label': 
            edges = self.scene_with_same_label_edges(no_of_nodes,node_labels)
        graph_lab = np.zeros((self.no_of_classes),dtype=float)
        graph_lab[act_class] = 1.0
        return combine_features,torch.tensor(np.array([graph_lab])),edges,is_act_start,is_act_end

# def custum_collate(batch):
#     scene_feats = []
#     agents = []
#     combine_features = []
#     edges = []
#     node_labels = []
#     no_of_nodes = []
#     act_class = []
#     is_act_start= []
#     is_act_end = []

#     for sample in batch:
#         scene_feats.append(torch.tensor(sample[0]))
#         agents.append(sample[1])
#         combine_features.append(torch.tensor(sample[2]))
#         edges.append(torch.tensor(sample[3]))
#         node_labels.append(sample[4])
#         no_of_nodes.append(sample[5])
#         act_class.append(torch.tensor(sample[6]))
#         is_act_start.append(sample[7])
#         is_act_end.append(sample[8])
        
#     return scene_feats,agents,torch.stack(combine_features,0),torch.stack(edges,0),node_labels,no_of_nodes,torch.stack(act_class,0),is_act_start,is_act_end

def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    no_of_nodes_per_graph = [0]
    num_nodes_seen = 0
    start_gt = []
    end_gt = []

    for features_labels_edge_index_tuple in batch:
        # print('node feat',features_labels_edge_index_tuple[0].shape)
        # print('edge_ind',features_labels_edge_index_tuple[2])
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        no_of_nodes_per_graph.append(features_labels_edge_index_tuple[0].shape[0])

        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        # print(features_labels_edge_index_tuple[0].shape[0])
        num_nodes_seen += features_labels_edge_index_tuple[0].shape[0]  # update the number of nodes we've seen so far
        start_gt.append(features_labels_edge_index_tuple[3])
        end_gt.append(features_labels_edge_index_tuple[4])
    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    start_gt = torch.tensor(start_gt)
    end_gt = torch.tensor(end_gt)
    edge_index = torch.cat(edge_index_list, 1)
    return node_features, node_labels, edge_index,no_of_nodes_per_graph,start_gt,end_gt