from enum import Enum

import tensorflow as tf
import cv2


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
    for fname in os.listdir(images_dir_path):
        val_image += [read_imgfile(os.path.join(images_dir_path, fname), w, h)]
    return val_image


def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s
