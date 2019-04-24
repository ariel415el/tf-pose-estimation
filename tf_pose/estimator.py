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


def fit_humans_to_size(humans,w, h):
    for human in humans:
        # human[::3] = (human[::3] * w + 0.5)
        # human[1::3] = (human[1::3] * h + 0.5)
        human[::3] = [int(x* w + 0.5) for x in human[::3]]
        human[1::3] = [int(x* h + 0.5) for x in human[1::3]]
    return humans




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
        return humans


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

        humans = fit_humans_to_size(humans, org_img.shape[1],org_img.shape[0])

        return humans


def create_debug_collage(inp, heatmap, vectmap, show_pose=False):
    global mplset
    mplset = True
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    peaksMap, upscaled_heatmap, upscaled_vectmaps = common.prepare_heatmaps(heatmap, vectmap, upsample=4)
    if show_pose:
        humans = PoseEstimator.estimate_paf(peaksMap, upscaled_heatmap, upscaled_vectmaps)
        humans = fit_humans_to_size(humans,inp.shape[1],inp.shape[0])
        debug_image = common.draw_humans(inp, humans, imgcopy=True)
        plt.imshow(debug_image)
    else:
        plt.imshow(inp)

    a = fig.add_subplot(2, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_heatmap.shape[1], upscaled_heatmap.shape[0])), alpha=0.5, cmap='gray')
    tmp = np.amax(upscaled_heatmap, axis=2)*255
    plt.imshow(tmp, cmap='hot', alpha=0.5)
    plt.colorbar()

    tmp2 = upscaled_vectmaps.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)*255
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)*255

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_vectmaps.shape[1], upscaled_vectmaps.shape[0])), alpha=0.5, cmap='gray')
    plt.imshow(tmp2_odd, cmap='hot', alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_vectmaps.shape[1], upscaled_vectmaps.shape[0])), alpha=0.5, cmap='gray')
    plt.imshow(tmp2_even, cmap='hot', alpha=0.5)
    plt.colorbar()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clear()
    plt.close()
    return data
