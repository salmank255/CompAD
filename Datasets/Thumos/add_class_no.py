import json


with open('Thumos_gt.json','r') as fff:
    anno_json = json.load(fff)

thumos_classes = ['Background',
'BaseballPitch',
 'BasketballDunk',
 'Billiards',
 'CleanAndJerk',
 'CliffDiving',
 'CricketBowling',
 'CricketShot',
 'Diving',
 'FrisbeeCatch',
 'GolfSwing',
 'HammerThrow',
 'HighJump',
 'JavelinThrow',
 'LongJump',
 'PoleVault',
 'Shotput',
 'SoccerPenalty',
 'TennisSwing',
 'ThrowDiscus',
 'VolleyballSpiking']


for vid in anno_json['database']:
    print(vid)
    for ii,segs in enumerate(anno_json['database'][vid]['annotations']):

        print(anno_json['database'][vid]['annotations'][ii]['label'])
        lab_ind = thumos_classes.index(anno_json['database'][vid]['annotations'][ii]['label'])
        print(lab_ind)
        anno_json['database'][vid]['annotations'][ii]['class_#'] = lab_ind

with open('Thumos_gt.json','w') as fff:
    json.dump(anno_json,fff)













