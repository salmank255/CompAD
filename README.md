# A Hybrid Graph Network for Complex Activity Detection in Video
***Salman Khan***, ***Izzeddin Teeti***, ***Andrew Bradley***, ***Mohamed Elhoseiny***, ***Fabio Cuzzolin***

[![arXiv](https://img.shields.io/badge/arXiv-Paper-FFF933)](https://arxiv.org/abs/2310.17493)

> **Abstract:** *Interpretation and understanding of video presents a challenging computer vision task in numerous fields - e.g. autonomous driving and sports analytics. Existing approaches to interpreting the actions taking place within a video clip are based upon Temporal Action Localisation (TAL), which typically identifies short-term actions. The emerging field of Complex Activity Detection (CompAD) extends this analysis to long-term activities, with a deeper understanding obtained by modelling the internal structure of a complex activity taking place within the video. We address the CompAD problem using a hybrid graph neural network which combines attention applied to a graph encoding the local (short-term) dynamic scene with a temporal graph modelling the overall long-duration activity. Our approach is as follows: i) Firstly, we propose a novel feature extraction technique which, for each video snippet, generates spatiotemporal `tubes' for the active elements (`agents') in the (local) scene by detecting individual objects, tracking them and then extracting 3D features from all the agent tubes as well as the overall scene. ii) Next, we construct a local scene graph where each node (representing either an agent tube or the scene) is connected to all other nodes. Attention is then applied to this graph to obtain an overall representation of the local dynamic scene. iii) Finally, all local scene graph representations are interconnected via a temporal graph, to estimate the complex activity class together with its start and end time. The proposed framework outperforms all previous state-of-the-art methods on all three datasets including ActivityNet-1.3, Thumos-14, and ROAD.*


<p align="center">
     <img src=./figs/framework.png > 
</p>



## Requirements
We need three things to get started with training: datasets, Feature extraction from the dataset, and pytorch with torchvision and tensoboardX. 

## Datasets

### ROAD

### ActivityNet-1.3

### Thumos-14

## Features Extraction
For all three dataset we already extracted the features and can be downloaded from here

## Training the model

```
python main.py --Data_Root=Datasets --Save_Root=Trained_models --Mode=train --Dataset_Name=Thumos --Seq_Len=30 --Edge_Type=scene --Inp_Feat_Len=2048 --No_of_Classes=21 --Aggregate_All=False --Epochs=500 --Train_Batch_Size=600 --Test_Batch_Size=600
```

### Arguments
```
--Data_Root=Datasets        --->  Path to the datasets where the features are saved
--Save_Root=Trained_models  --->  Path for saving the trained models
--Mode=train                --->  train or test
--Dataset_Name=ROAD         --->  Datasets are ROAD, Activitynet, and Thumos        
--Seq_Len=30                --->  Sequence supported by our system are 12, 18, 24, and 30
--Edge_Type=fully_connected --->  Three types of edges; fully_connected scene scene_same_label
--Inp_Feat_Len=2048         --->  For our features the len is 2048, please change it if you are using your own features
--No_of_Classes=7           --->  No_of_classes are 7,21, and 200 in ROAD, Thumos, and ActivityNet
--Aggregate_All=False       --->  Using the agreegation of all nodes or only scene node
--Epochs=500                --->  Number of epochs
--Train_Batch_Size=600      --->  Length of the temporal graph at training time
--Test_Batch_Size=600       --->  Length of the temporal grpah at test time
```

## Visualisation
<!-- 
![Tracking and scene graph](./figs/tracking_scene_g.png)

![Visual results](./figs/qaul_res.png) -->


## Citation

```bibtex
@InProceedings{khan2023hybrid,
      title={A Hybrid Graph Network for Complex Activity Detection in Video}, 
      author={Salman Khan and Izzeddin Teeti and Andrew Bradley and Mohamed Elhoseiny and Fabio Cuzzolin},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      year={2024}
}
```
