import os
import sys
import time
import argparse
import logging
import math
import gc
import json

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord
MXNET_CUDNN_AUTOTUNE_DEFAULT = 0

class Feat_Extractor:
    def __init__(self):
        self.opt = self.parse_args()
        gc.set_threshold(100, 5, 5)

        # set env
        if self.opt.gpu_id == -1:
            self.context = mx.cpu()
        else:
            gpu_id = self.opt.gpu_id
            self.context = mx.gpu(gpu_id)

        # get data preprocess
        image_norm_mean = [0.485, 0.456, 0.406]
        image_norm_std = [0.229, 0.224, 0.225]
        if self.opt.ten_crop:
            self.transform_test = transforms.Compose([
                video.VideoTenCrop(self.opt.input_size),
                video.VideoToTensor(),
                video.VideoNormalize(image_norm_mean, image_norm_std)
            ])
            self.opt.num_crop = 10
        elif self.opt.three_crop:
            self.transform_test = transforms.Compose([
                video.VideoThreeCrop(self.opt.input_size),
                video.VideoToTensor(),
                video.VideoNormalize(image_norm_mean, image_norm_std)
            ])
            self.opt.num_crop = 3
        else:
            self.transform_test = video.VideoGroupValTransform(size=self.opt.input_size, mean=image_norm_mean, std=image_norm_std)
            self.opt.num_crop = 1


        # get model
        if self.opt.use_pretrained and len(self.opt.hashtag) > 0:
            self.opt.use_pretrained = self.opt.hashtag
        classes = self.opt.num_classes
        model_name = self.opt.model
        self.net = get_model(name=model_name, nclass=classes, pretrained=self.opt.use_pretrained,
                        feat_ext=True, num_segments=self.opt.num_segments, num_crop=self.opt.num_crop)
        self.net.cast(self.opt.dtype)
        self.net.collect_params().reset_ctx(self.context)
        if self.opt.mode == 'hybrid':
            self.net.hybridize(static_alloc=True, static_shape=True)
        if self.opt.resume_params != '' and not self.opt.use_pretrained:
            self.net.load_parameters(self.opt.resume_params, ctx=self.context)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Extract features from pre-trained models for video related tasks.')
        parser.add_argument('--data-dir', type=str, default='',
                            help='the root path to your data')
        parser.add_argument('--need-root', action='store_true',
                            help='if set to True, --data-dir needs to be provided as the root path to find your videos.')
        parser.add_argument('--data-list', type=str, default='',
                            help='the list of your data. You can either provide complete path or relative path.')
        parser.add_argument('--dtype', type=str, default='float32',
                            help='data type for training. default is float32')
        parser.add_argument('--gpu-id', type=int, default=0,
                            help='number of gpus to use. Use -1 for CPU')
        parser.add_argument('--mode', type=str,
                            help='mode in which to train the model. options are symbolic, imperative, hybrid')
        parser.add_argument('--model', type=str, default='i3d_resnet50_v1_kinetics400', 
                            help='type of model to use. see vision_model for options.')
        parser.add_argument('--input-size', type=int, default=224,
                            help='size of the input image size. default is 224')
        parser.add_argument('--use-pretrained', action='store_true', default=True,
                            help='enable using pretrained model from GluonCV.')
        parser.add_argument('--hashtag', type=str, default='',
                            help='hashtag for pretrained models.')
        parser.add_argument('--resume-params', type=str, default='',
                            help='path of parameters to load from.')
        parser.add_argument('--log-interval', type=int, default=10,
                            help='Number of batches to wait before logging.')
        parser.add_argument('--new-height', type=int, default=256,
                            help='new height of the resize image. default is 256')
        parser.add_argument('--new-width', type=int, default=340,
                            help='new width of the resize image. default is 340')
        parser.add_argument('--new-length', type=int, default=32,
                            help='new length of video sequence. default is 32')
        parser.add_argument('--new-step', type=int, default=1,
                            help='new step to skip video sequence. default is 1')
        parser.add_argument('--num-classes', type=int, default=400,
                            help='number of classes.')
        parser.add_argument('--ten-crop', action='store_true',
                            help='whether to use ten crop evaluation.')
        parser.add_argument('--three-crop', action='store_true',
                            help='whether to use three crop evaluation.')
        parser.add_argument('--video-loader', action='store_true', default=True,
                            help='if set to True, read videos directly instead of reading frames.')
        parser.add_argument('--use-decord', action='store_true', default=True,
                            help='if set to True, use Decord video loader to load data.')
        parser.add_argument('--slowfast', action='store_true',
                            help='if set to True, use data loader designed for SlowFast network.')
        parser.add_argument('--slow-temporal-stride', type=int, default=16,
                            help='the temporal stride for sparse sampling of video frames for slow branch in SlowFast network.')
        parser.add_argument('--fast-temporal-stride', type=int, default=2,
                            help='the temporal stride for sparse sampling of video frames for fast branch in SlowFast network.')
        parser.add_argument('--num-crop', type=int, default=1,
                            help='number of crops for each image. default is 1')
        parser.add_argument('--data-aug', type=str, default='v1',
                            help='different types of data augmentation pipelines. Supports v1, v2, v3 and v4.')
        parser.add_argument('--num-segments', type=int, default=1,
                            help='number of segments to evenly split the video.')
        parser.add_argument('--save-dir', type=str, default='./',
                            help='directory of saved results')
        opt = parser.parse_args()
        return opt

    def read_data(self,opt, clip_input, transform):
        clip_input = transform(clip_input)
        if opt.slowfast:
            sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        else:
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        if opt.new_length == 1:
            clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

        return nd.array(clip_input)

    def process(self,inp_list):
        self.opt.new_length = len(inp_list)
        inp_data = self.read_data(self.opt,inp_list,self.transform_test)
        video_input = inp_data.as_in_context(self.context)
        video_feat = self.net(video_input.astype(self.opt.dtype, copy=False))
        return video_feat.asnumpy()
           