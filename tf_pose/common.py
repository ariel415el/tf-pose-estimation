from enum import Enum
import tensorflow as tf
import cv2
import numpy as np
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


def get_sample_images(w, h):
    import os
    val_image = []
    images_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../images"
    if not os.path.exists(images_dir_path):
            images_dir_path = "./images"   
    for fname in os.listdir(images_dir_path):
        val_image += [read_imgfile(os.path.join(images_dir_path, fname), w, h)]
    return val_image


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


def draw_humans(npimg, humans, imgcopy=False):
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
        v = kp[2::3]
        random_color = list(np.random.rand(1,3)[0]*255)
        # draw limbs
        for sk in sks:
            if np.all(v[sk] > 0):
                center_1 = (x[sk[0]],y[sk[0]])
                center_0 = (x[sk[1]],y[sk[1]])
                cv2.line(npimg, center_0, center_1, random_color, 3)
        for i in range(BC_parts.Background.value):
            if v[i] > 0:
                cv2.circle(npimg, (x[i], y[i]), 3, [0,0,0], thickness=3, lineType=8, shift=0)

    return npimg