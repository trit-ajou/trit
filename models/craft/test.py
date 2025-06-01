"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utils, imgproc, file_utils
import json
import zipfile

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
#
# args = parser.parse_args()




result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):


    # resize ###arg.size = 1280으로 바꿈(default), mag_ratio=1.5(default)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # print("SCore text-------------------------------")
    # print(score_text)
    # print("--------------------------------------")

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # print("boxes----------------------------")
    # print(boxes)
    # print("-----------------------------------")
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text