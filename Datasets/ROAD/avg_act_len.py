import json
from statistics import mean

with open('ROAD_gt.json', 'r') as f:
    road_json = json.load(f)   


act_lens = []

for vid in road_json['database']:
    for anno in road_json['database'][vid]['annotations']:
        act_lens.append(anno['segment'][1]-anno['segment'][0])
        print(vid,anno['segment'][1]-anno['segment'][0])

print(act_lens)
print(mean(act_lens)/12)








