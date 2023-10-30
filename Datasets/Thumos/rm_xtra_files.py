import json
import shutil
import os
with open('Thumos_gt.json','rb') as ff:
    th_json = json.load(ff)

for item in th_json['database']:
    print(item)
    dest = shutil.move('rgb-images_/'+item, 'rgb-images/'+item)


