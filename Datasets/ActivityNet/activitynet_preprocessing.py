import json
import shutil
import os
with open('activity_net.v1-3.min.json','rb') as ff:
    act_json = json.load(ff)


# for item in act_json['database']:
#     subset = act_json['database'][item]['subset']
#     if subset == "testing":
#         dest = shutil.move('videos/v_'+item+'.mp4', 'testing_vids/v_'+item+'.mp4')
#         print(item)
#         # break

# print(os.path.isdir('rgb-images/v_'+'xZEl3yh0Cos'+'/'))
for item in act_json['database']:
    subset = act_json['database'][item]['subset']
    if subset == "testing":
        isdir = os.path.isdir('rgb-images/v_'+item+'/')
        if isdir:
            dest = shutil.move('rgb-images/v_'+item+'/', 'testing_frames/v_'+item+'/')
        # print(isdir)
        # if isdir:
        #     dest = shutil.move('rbg-images/v_'+item, 'testing_frames/v_'+item)
        # print(item)

# print(len(act_json['database']))