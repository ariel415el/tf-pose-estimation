from enum import Enum
import tensorflow as tf
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

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


def get_sample_images(anns_path, w, h):
    anns = json.load(open(anns_path))

    resized_anns = []
    resized_images = []
    for k in anns:
        image = cv2.imread(k, cv2.IMREAD_COLOR)
        resized_images += [cv2.resize(image, (w, h))]

        x_factor = w / image.shape[1]
        y_factor = h / image.shape[0]
        resized_sets = []
        for ref_set in anns[k]['keypoint_sets']:
            new_set = ref_set.copy()
            new_set[::3]  = [x*x_factor for x in ref_set[::3]]
            new_set[1::3] = [y * y_factor for y in ref_set[1::3]]
            if new_set != [] and np.sum(new_set[::3]) != 0 and np.sum(new_set[1::3]) != 0:
                resized_sets += [new_set]
        resized_anns += [resized_sets]

    anns = [anns[k]['keypoint_sets'] for k in anns]
    return resized_images, resized_anns


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


def draw_humans(npimg, humans, color=None, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]

    colors = np.random.rand(1,3)
    sks = np.array([(0,1),(1,2),(1,3), (2,4),(4,6), (3,5),(5,7), (1,8),(8,10),(10,12), (1,9),(9,11),(11,13)])
    centers = {}
    
    for i,kp in enumerate(humans):
        # draw point
        x = kp[0::3]
        y = kp[1::3]
        v = np.array(kp[2::3])
        if color is not None:
         person_color = color
        else:
            person_color = list(np.random.rand(1,3)[0]*255)
        # draw limbs
        for sk in sks:
            if np.all(v[sk] > 0):
                center_1 = (int(x[sk[0]] + 0.5), int(y[sk[0]] + 0.5))
                center_0 = (int(x[sk[1]] + 0.5), int(y[sk[1]] + 0.5))
                cv2.line(npimg, center_0, center_1, person_color, 2)
        for i in range(BC_parts.Background.value):
            if v[i] > 0:
                cv2.circle(npimg, (int(x[i] + 0.5), int(y[i] + 0.5)), 1, [0,0,0], thickness=1, lineType=4, shift=0)

    return npimg

def plot_from_csv(input_path, output_path, plotees):
    from itertools import cycle
    cycol = cycle('bgrcmk')

    import pandas as pd
    pd = pd.read_csv(input_path)
    step_nums = np.array(pd['Step_number'])
    for data_name in plotees:
        plt.plot(step_nums,  np.array(pd[data_name]), label=data_name, c=next(cycol))
    plt.xlabel('Step_number')
    plt.legend(loc='upper left')
    plt.savefig(output_path)
    plt.close()