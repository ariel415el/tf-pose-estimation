import logging
import math
import os
import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import BC_pairs
from tf_pose.tensblur.smoother import Smoother
from tf_pose.postProcess  import python_paf_process
from tf_pose.postProcess.python_paf_process import NUM_PART, NUM_HEATMAP

def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        subset, flat_info =  python_paf_process.estimate_paf(peaks, heat_mat, paf_mat)
        humans = []
        for human_id in range(len(subset)):
            human = []
            is_added = False

            for part_idx in range(NUM_PART):
                c_idx = int(subset[human_id][part_idx])
                if c_idx < 0:
                    human += [0, 0, 0]
                else:
                    x = float(flat_info[c_idx]._peakX) / heat_mat.shape[1]
                    y = float(flat_info[c_idx]._peakY) / heat_mat.shape[0]
                    score = 2 # flat_info[c_idx]._paekScore
                    human += [x, y, score]


            humans += [human]
        return np.array(humans)


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, input_name, output_name, input_size=(432, 368), tf_config=None, numHeatMaps=len(common.BC_parts)):
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # load graph
        print("Initializing TfPoseEstimator:")
        print('\tLoading graph from %s(default size=%dx%d)' % (graph_path, self.input_width, self.input_height))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/%s'%input_name)
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/%s'%output_name)
        self.tensor_heatMat = self.tensor_output[:, :, :, :numHeatMaps]
        self.tensor_pafMat = self.tensor_output[:, :, :, numHeatMaps:]
        self.workingHeatMatSize = tf.placeholder(dtype=tf.int32, shape=(2,), name='workingHeatMatSize')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :numHeatMaps], self.workingHeatMatSize,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, numHeatMaps:], self.workingHeatMatSize,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None
        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        print("\tWarm up graph")
        for i in [1,2,4,8]:  
            self.persistent_sess.run(
                [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
                feed_dict={
                    self.tensor_image: [np.ndarray(shape=(self.input_height, self.input_width, 3), dtype=np.float32)],
                    self.workingHeatMatSize: [self.input_height // i, self.input_width // i]
                }
            )
        # logs
        if self.tensor_image.dtype == tf.quint8:
            print('\tquantization mode enabled.')

    def __del__(self):
        # self.persistent_sess.close()
        pass

    def get_flops(self):
        flops = tf.profiler.profile(self.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        return flops.total_float_ops

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
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
            for i in range(common.BC_parts.Background.value):
                if v[i] > 0:
                    cv2.circle(npimg, (x[i], y[i]), 3, [0,0,0], thickness=3, lineType=8, shift=0)

        return npimg



    def inference(self, org_img, upsampleHeatMaps):
        if org_img is None:
            raise Exception('The image is not valid. Please check your image exists.')
        npimg = cv2.resize(org_img, (self.input_width, self.input_height))    
        output_dimentions = np.array([self.input_height, self.input_width]) / 8


        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass
        t = time.time()
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], 
            feed_dict={self.tensor_image: [npimg], self.workingHeatMatSize: output_dimentions * upsampleHeatMaps}
            )
        print('\tInference time=%.5f' % (time.time() - t))
        print ("\tInference input shape: ", npimg.shape)
        print ("\tInference workingHeatMatSize shape: ", output_dimentions * upsampleHeatMaps)
        print ("\tInference output heatmaps shape: ", heatMat_up.shape)
        print ("\tInference output pafs shape: ", pafMat_up.shape)
        print ("\tInference output peaks shape: ", peaks.shape)

        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        print('\tPost-process time=%.5f' % (time.time() - t))
    
        # fit humans to original image size
        for human in humans:
            human[::3] = (human[::3]*org_img.shape[1] +0.5)
            human[1::3] = (human[1::3]*org_img.shape[0] +0.5)

        return humans.astype(int)

