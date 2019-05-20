import os
import cv2
import numpy as np
import tensorflow as tf
import time
import sys
import common
from tensblur.smoother import Smoother
from postProcess  import python_paf_process
from postProcess.python_paf_process import NUM_PART, NUM_HEATMAP
import debug_tools

import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, parent_dir)
from evaluation.BCEvaluation import filter_anns, computeOks, compute_roc

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
                    score = float(flat_info[c_idx]._paekScore)
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

def evaluate_results(images, list_heatmaps, list_pafmaps, imgs_paths, gt_anns_list):
    iou_threshold = 0.5
    tps = 0
    gts = 0
    dts = 0
    collages = []
    accuracy_images = []
    det_kps_dict = {}
    gt_kps_dict = {}
    for i in range(len(images)):
        image = common.get_bgimg(images[i]).copy()
        peaksMap, upscaled_heatmap, upscaled_vectmaps = common.prepare_heatmaps(list_heatmaps[i], list_pafmaps[i], upsample=4)
        humans = PoseEstimator.estimate_paf(peaksMap, upscaled_heatmap, upscaled_vectmaps)
        humans = fit_humans_to_size(humans, image.shape[1], image.shape[0])

        debgug_collage = debug_tools.create_debug_collage(image, upscaled_heatmap, upscaled_vectmaps, humans)

        debgug_collage = cv2.resize(debgug_collage, (640, 640))
        debgug_collage = debgug_collage.reshape([640, 640, 3]).astype(float)
        collages += [debgug_collage]
        det_kps_dict[imgs_paths[i]] = humans
        gt_kps_dict[imgs_paths[i]] = gt_anns_list[i]

        clean_humans, _  = filter_anns(humans, vis_th=0, min_kps=3, min_height=40, ignore_head=False)
        debug_tools.draw_humans(image, np.array(gt_anns_list[i]), color=[0, 255, 0])
        debug_tools.draw_humans(image, clean_humans, color=[255, 0, 0])
        cv2.putText(image,"GTs: %d"%len(gt_anns_list[i]), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, 2555)
        dts += len(clean_humans)
        gts += len(gt_anns_list[i])
        if len(clean_humans) > 0 and len(gt_anns_list[i]) > 0:
            ious = computeOks(np.array(gt_anns_list[i]), clean_humans).transpose()  # this is a (gt)x(det) matrices of ious
            first_matches_idxs = np.argmax(ious, axis=1)  # best det for each gt
            first_matches_vals = ious[np.arange(ious.shape[0]), first_matches_idxs]
            valid_matches_indices = np.where(first_matches_vals > iou_threshold)[0]  # best det for each gt
            tps += len(valid_matches_indices)
            cv2.putText(image,"TPs: %d"%len(valid_matches_indices), (10,60), cv2.FONT_HERSHEY_PLAIN, 2, 255)
        accuracy_images += [image.astype(np.float32)]
    visThrs, recall_vals, precision_vals, accuracy_images = compute_roc(gt_kps_dict, det_kps_dict,
        iou_th=iou_threshold,
        gt_vis_th=1,
        min_kps=3,
        min_height=50,
        ignore_head=False,
        debug_vis_th=0.2) 
         
        
    # recall, precision, debug_images_set = evaluate(list_gt_keypoint_sets,
    #                         det_kps_dict,
    #                         iou_threshold,
    #                         get_debug_image=debug)

    # Compute recall precision
    if gts == 0:
        recall = 1
    else:
        recall = min(1, max(0, tps / float(gts)))
    if dts == 0 :
        precision = 1
    else:
        precision = min(1, max(0, tps / float(dts))) # tps may be larger than dets

    return recall, precision, collages, accuracy_images