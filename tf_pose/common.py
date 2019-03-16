from enum import Enum

import tensorflow as tf
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu



class BCJtaPart(Enum):
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

BCJtaFlippedParts = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12 ,14]
BCJtaPairs = [(0,1),(1,2),(1,3), (2,4),(4,6), (3,5),(5,7), (2,8),(8,10),(10,12), (3,9),(9,11),(11,13)]


class OpenPosePart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18
    
OpenPoseFlippedParts = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18]
OpenPosePairs = [(2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14), (2, 3), (3, 4),
             (4, 5), (3, 17), (2, 6), (6, 7), (7, 8), (6, 18), (2, 1), (1, 15), (1, 16), (15, 17), (16, 18)]

OpenPosePairsRender = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                     (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


class MPIIPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    RWrist = 6
    RElbow = 7
    RShoulder = 8
    LShoulder = 9
    LElbow = 10
    LWrist = 11
    Neck = 12
    Head = 13

    @staticmethod
    def from_coco(human):
        t = [
            (MPIIPart.Head, OpenPosePart.Nose),
            (MPIIPart.Neck, OpenPosePart.Neck),
            (MPIIPart.RShoulder, OpenPosePart.RShoulder),
            (MPIIPart.RElbow, OpenPosePart.RElbow),
            (MPIIPart.RWrist, OpenPosePart.RWrist),
            (MPIIPart.LShoulder, OpenPosePart.LShoulder),
            (MPIIPart.LElbow, OpenPosePart.LElbow),
            (MPIIPart.LWrist, OpenPosePart.LWrist),
            (MPIIPart.RHip, OpenPosePart.RHip),
            (MPIIPart.RKnee, OpenPosePart.RKnee),
            (MPIIPart.RAnkle, OpenPosePart.RAnkle),
            (MPIIPart.LHip, OpenPosePart.LHip),
            (MPIIPart.LKnee, OpenPosePart.LKnee),
            (MPIIPart.LAnkle, OpenPosePart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for mpi, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty

def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(w, h):
    import os
    val_image = []
    for fname in os.listdir("./images"):
        val_image += [read_imgfile(os.path.join("./images", fname), w, h)]
    return val_image


def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s
