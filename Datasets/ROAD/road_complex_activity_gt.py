import pandas as pd 
from pandas import ExcelWriter
from pandas import ExcelFile 
from agentDetection import AutoDict
import json

gt_json = AutoDict()
video_names = []
DataF=pd.read_excel("Complex_activity_annotation.xlsx",sheet_name='Sheet1')
print(DataF.columns)
for i in range(len(DataF)):
    video_name = DataF.loc[i]['V_Name']
    video_names.append(video_name)
    if i ==0:
        gt_json['database'][video_name]['subset'] = DataF.loc[i]['Subset']
        gt_json['database'][video_name]['annotations']= [{"segment":[int(DataF.loc[i]['Start_Frame']),int(DataF.loc[i]['End_Frame'])],"label":DataF.loc[i]['Class_names'],"class_#":int(DataF.loc[i]['Class_#'])}]
    else:
        if video_name == video_names[i-1]:
            gt_json['database'][video_name]['annotations'].append({"segment":[int(DataF.loc[i]['Start_Frame']),int(DataF.loc[i]['End_Frame'])],"label":DataF.loc[i]['Class_names'],"class_#":int(DataF.loc[i]['Class_#'])})
        else:
            gt_json['database'][video_name]['subset'] = DataF.loc[i]['Subset']
            gt_json['database'][video_name]['annotations']= [{"segment":[int(DataF.loc[i]['Start_Frame']),int(DataF.loc[i]['End_Frame'])],"label":DataF.loc[i]['Class_names'],"class_#":int(DataF.loc[i]['Class_#'])}]


with open('ROAD_Complex_gt.json', 'w') as outfile:
        json.dump(gt_json, outfile)



