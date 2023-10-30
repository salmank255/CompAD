import sys
sys.path.insert(0, 'Yolov5_DeepSort_Pytorch')
sys.path.insert(0, 'Yolov5_DeepSort_Pytorch/yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json
import glob
import random
import string
import numpy as np

def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str


def detect(opt,inp_data,inp_org):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # Inference
    t1 = time_sync()
    inp_data = inp_data.half() if half else inp_data.float()  # uint8 to fp16/32
    # inp_data /= 255.0  # 0 - 255 to 0.0 - 1.0
    pred = model(inp_data, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_sync()
    # print(inp_data[0].shape)
    # print(rr)
    # Process detections
    rois = []
    # print(len(pred))
    # print(rr)
    for i, det in enumerate(pred):  # detections per image
        # print('i',i)

        if det is not None and len(det):
            # print(inp_org[i].detach().numpy().shape)
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                inp_data.shape[2:], det[:, :4], inp_org[i].detach().numpy().shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            # pass detections to deepsort
            # print(inp_org[i].detach().numpy().shape)
            # print(rr)
            # print(xywhs.cpu())
            # print(confs.cpu())
            # print(clss.cpu())
            immm = inp_org[i].detach().numpy()
            outputss = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), inp_org[i].detach().numpy())
            if len(outputss) > 0:
                for outputs in outputss:
                    cv2.rectangle(immm, (int(outputs[0]), int(outputs[1])), (int(outputs[2]), int(outputs[3])), (0, 255, 0), 2)
                    cv2.putText(immm, str(outputs[4]), (int(outputs[0]), int(outputs[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
                    cv2.putText(immm, str(outputs[5]), (int(outputs[0]), int(outputs[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.imwrite('temp_outs/'+str(i)+'_.png',immm)
            # print('outputs',outputs)
            
        else:
            deepsort.increment_ages()
            outputss = []
        rois.append(outputss)
    # print(rr)
    return rois        
        
def process(inp_data,inp_org):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='Yolov5_DeepSort_Pytorch/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='Yolov5_DeepSort_Pytorch/inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="Yolov5_DeepSort_Pytorch/deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    #### classes to target
    # args.classes = [0,1,2,4,6,9,]
    # print(args.classes)
    with torch.no_grad():
        rois = detect(args,inp_data,inp_org)
    return rois