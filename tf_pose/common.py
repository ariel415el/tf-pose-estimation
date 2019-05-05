from enum import Enum
import tensorflow as tf
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from numpy import random
regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

class BC_parts(Enum):
    head_center = 0
    neck = 1
    left_shoulder = 2
    right_shoulder = 3
    left_elbow = 4
    right_elbow = 5
    left_wrist = 6
    right_wrist = 7
    left_hip = 8
    right_hip = 9
    left_knee = 10
    right_knee = 11
    left_ankle = 12
    right_ankle = 13
    Background = 14

BC_flipped_indices = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12 ,14]
BC_pairs = [(0,1),(1,2),(1,3), (2,4),(4,6), (3,5),(5,7), (1,8),(8,10),(10,12), (1,9),(9,11),(11,13)]

def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(anns_path, w, h, batchsize, subsample=None):
    anns = json.load(open(anns_path))
    size = batchsize if subsample is None else max(batchsize, min(subsample,len(anns)))
    size = (size // batchsize)*batchsize
    
    anns = {k:anns[k] for k in random.choice(list(anns.keys()), size) }

    resized_anns_dict = {}
    for k in anns:
        # image = cv2.imread(k, cv2.IMREAD_COLOR)
        # x_factor = w / image.shape[1]
        # y_factor = h / image.sh
        x_factor = w / anns[k]['img_width']
        y_factor = h / anns[k]['img_height']
        resized_sets = []
        for ref_set in anns[k]['keypoint_sets']:
            new_set = ref_set.copy()
            new_set[::3]  = [x*x_factor for x in ref_set[::3]]
            new_set[1::3] = [y * y_factor for y in ref_set[1::3]]
            if new_set != [] and np.sum(new_set[::3]) != 0 and np.sum(new_set[1::3]) != 0:
                resized_sets += [new_set]
        resized_anns_dict[k] = resized_sets

    return resized_anns_dict


def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s

def get_bgimg(inp, target_size=None):
    inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp

def get_grey_img(inp, target_size=None):
    inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp


def prepare_heatmaps(raw_heatmap, raw_vectmap, upsample=4):
    h = raw_vectmap.shape[0]*upsample
    w = raw_vectmap.shape[1]*upsample
    upscaled_heatmap = cv2.resize(raw_heatmap,(h, w))
    upscaled_vectmaps = cv2.resize(raw_vectmap,(h, w))
    smoothed_heatmap =  cv2.GaussianBlur(upscaled_heatmap,(25,25),3,3)
    dilated_smoothed_heatmap = cv2.dilate(smoothed_heatmap, None)
    peaksMap = np.where(smoothed_heatmap == dilated_smoothed_heatmap, smoothed_heatmap, 0)
    return peaksMap, upscaled_heatmap, upscaled_vectmaps
