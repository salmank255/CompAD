import cv2
import glob
import os
import math
from Yolov5_DeepSort_Pytorch.process_vis import process
import time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl

from PIL import Image
 
# def fig2data ( fig ):
#     """
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     # draw the renderer
#     fig.canvas.draw ( )
 
#     # Get the RGBA buffer from the figure
#     w,h = fig.canvas.get_width_height()
#     buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
#     buf.shape = ( w, h,4 )
 
#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = np.roll ( buf, 3, axis = 2 )
#     return buf

# def fig2img ( fig ):
#     """
#     @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
#     @param fig a matplotlib figure
#     @return a Python Imaging Library ( PIL ) image
#     """
#     # put the figure pixmap into a numpy array
#     buf = fig2data ( fig )
#     w, h, d = buf.shape
#     return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe','backpack', 'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
               'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
               'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
               'chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
               'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

def set_video(inp,video_graph_p,video_out_p,video_cat_p):
    cap = cv2.VideoCapture(inp)
    _, frame = cap.read()
    frheight, frwidth, ch = frame.shape
    fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_graph = cv2.VideoWriter(video_graph_p, fourcc, fps, (frwidth, frheight))
    video_out = cv2.VideoWriter(video_out_p, fourcc, fps, (frwidth, frheight))
    video_cat = cv2.VideoWriter(video_cat_p, fourcc, fps, (frwidth*2, frheight))
    return cap,video_graph,video_out,video_cat, fps,total_frames,frwidth,frheight

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

Dataset_name = 'waymo'
output_path = 'waymo_vis_results'

for file_path in glob.glob(Dataset_name+'/*.mp4'):
    inp_video_path = os.path.splitext(os.path.basename(file_path))[0]
    # inp_video_path = '2014-06-25-16-45-34_stereo_centre_02_1'
    print('Video_name: ',inp_video_path)
    
    inp = Dataset_name+'/'+inp_video_path+'.mp4'
    video_graph_p = output_path+'/'+inp_video_path+'_graph.mp4'
    video_out_p = output_path+'/'+inp_video_path+'_out.mp4'
    video_cat_p = output_path+'/'+inp_video_path+'_concat.mp4'
    cap,video_graph,video_out,video_cat,fps,total_frames,frwidth,frheight = set_video(inp,video_graph_p,video_out_p,video_cat_p)
    print('Total_frames: ',total_frames)
    rois = process(inp)
    # print(len(rois))

    roi = rois[3]

    print(roi)
    print(rr)
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure(figsize=(19.2*2,12.8*2))
    ax = fig.add_subplot(111,projection='3d')


    """                                                                                                                                                    
    Scaling is done from here...                                                                                                                           
    """
    x_scale=1
    y_scale=2
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0



    ax.get_proj=short_proj

    cap.set(1,0)
    for f_no in range(len(rois)):
        
        print(f_no)
        roi = rois[f_no]
        if roi == []:
            ret,inp_img = cap.read()
            continue
        ret,inp_img = cap.read()
        for rr in roi:

            # r = [-1,1]
            # X, Y = np.meshgrid(r, r)
            # ax.scatter3D(np.arange(frwidth), np.arange(frwidth), np.arange(frwidth), alpha = .0)
            np.random.seed(rr[4])
            color = list(np.random.rand(3))
            
            bottom_left = [rr[0],f_no,rr[3]]
            bottom_right = [rr[2],f_no,rr[3]]
            top_right = [rr[2],f_no,rr[1]]
            top_left =[rr[0],f_no,rr[1]]
            

            
            verts = [[bottom_left,bottom_right,top_right,top_left]]
            ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors=color, alpha=.0))
            ax.set_xlim([0,frwidth])
            ax.set_ylim([0,total_frames])
            ax.set_zlim([0,frheight])

            color = [i * 255 for i in color]

            cv2.rectangle(inp_img, (int(top_left[0]), int(top_left[2])), (int(bottom_right[0]), int(bottom_right[2])), color, 2)
            cv2.putText(inp_img, coco_labels[rr[5]], (int(top_left[0]), int(top_left[2]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(inp_img, str(rr[4]), (int(top_left[0]), int(top_left[2]-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        
        # ax.view_init(240, 45)
        ax.view_init(-140, 60)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        plt.show()
        plt.savefig('waymofoo.png')
        graph_img_ = cv2.imread('waymofoo.png')
        graph_img = graph_img_[560:1900,1575:2980,:]
        graph_img = cv2.resize(graph_img,(1920,1280))
        video_graph.write(graph_img)
        video_out.write(inp_img)
        video_cat.write(np.uint8(np.concatenate((inp_img, graph_img), axis=1)))
        # time.sleep(3)
    cap.release()
    video_graph.release()
    video_out.release()
    video_cat.release()
    # break