import glob
import json
import cv2

with open('Thumos_gt.json','r') as fff:
    anno_json = json.load(fff)

for vid in anno_json['database']:
    print(vid)
    video = cv2.VideoCapture('videos/'+vid+'.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    for ii,segs in enumerate(anno_json['database'][vid]['annotations']):
        
        seg_new = [int(float(segs['segment'][0])*fps),int(float(segs['segment'][1])*fps)]
        segs['segment'] = seg_new

with open('Thumos_gt.json','w') as fff:
    json.dump(anno_json,fff)













