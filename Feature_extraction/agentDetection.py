

from numpy.lib import utils
from Yolov5_DeepSort_Pytorch.process import process
import cv2 
import numpy as np

def get_coco_class_list():
    return ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
            'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
            'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
            'cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
            'book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

class AutoDict(dict):
    def __missing__(self, k):
        self[k] = AutoDict()
        return self[k]

def detect_Agent(inp,inp_org,Dataset_name):
    ### To be updated for parallel processing
    ### To be updated for loading once
    agent_tubes = []
    for batch in range(inp.shape[0]):
        # print(inp[batch].shape)
        # print(inp_org[batch].shape)
        tubes = process(inp[batch],inp_org[batch],Dataset_name)
        refined_tubes,unique_tubes_update = check_for_connected_boxes(tubes)
        agent_tubes.append(refined_tubes)
    return agent_tubes

def check_for_connected_boxes(tubes):
    refined_tubes = AutoDict()
    unique_tubes = set()
    tubes[0] = tubes[2]
    tubes[1] = tubes[2]
    for i,tube in enumerate(tubes):
        for bbox in tube:
            refined_tubes[bbox[4]]['class'] = bbox[5]
            refined_tubes[bbox[4]][i] = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[5]]
            unique_tubes.add(bbox[4])
    unique_tubes_update = unique_tubes.copy()
    remove_tube_flag = False
    for u_tube in unique_tubes:
        min_lens = []
        for i,tt in enumerate(refined_tubes[u_tube]):
            if tt == 'class':
                continue
            min_lens.append(min(refined_tubes[u_tube][tt][2]-refined_tubes[u_tube][tt][0],refined_tubes[u_tube][tt][3]-refined_tubes[u_tube][tt][1]))
            if refined_tubes[u_tube][tt][0] > refined_tubes[u_tube][tt][2] or refined_tubes[u_tube][tt][1] > refined_tubes[u_tube][tt][3]:
                remove_tube_flag = True
                break
        frame_min_len = min(min_lens)
        if len(refined_tubes[u_tube]) < 6 or frame_min_len < 12 or remove_tube_flag:
            remove_tube_flag = False
            del refined_tubes[u_tube]
            unique_tubes_update.remove(u_tube)
    return refined_tubes,unique_tubes_update


def cropping_Agents(agents_tubes,image_list):
    cropped_agents = AutoDict()
    coco_list = get_coco_class_list()
    for agent in agents_tubes:
        agent_cropped_frames = []
        for f_bbox in agents_tubes[agent]:
            if f_bbox == 'class':
                # print(coco_list[agents_tubes[agent][f_bbox]])
                class_label = coco_list[agents_tubes[agent][f_bbox]]
                class_index = agents_tubes[agent][f_bbox]
                continue
            frame1 = image_list[f_bbox][agents_tubes[agent][f_bbox][1]:agents_tubes[agent][f_bbox][3],agents_tubes[agent][f_bbox][0]:agents_tubes[agent][f_bbox][2],:]
            # print(image_list[f_bbox].shape)
            # print(agents_tubes[agent][f_bbox][1],agents_tubes[agent][f_bbox][3],agents_tubes[agent][f_bbox][0],agents_tubes[agent][f_bbox][2])
            # print(frame1.shape)
            frame1 = cv2.resize(frame1,(224,224))
            agent_cropped_frames.append(frame1)
            # cv2.imwrite('temp_outs/'+str(agent)+str(f_bbox)+'out.png',np.uint8(frame1))
        cropped_agents[agent] = {"class_label":class_label,"class_index":class_index,"cropped_frames":agent_cropped_frames}
    
    return cropped_agents
