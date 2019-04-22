import logging
import math
import multiprocessing
import struct
import sys
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import random
import requests
import cv2
import numpy as np
import time

import tensorflow as tf

from tensorpack.dataflow import MultiThreadMapData
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.parallel import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from tf_pose.pose_augment import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale
import tf_pose.common as common
from numba import jit
import json 

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mplset = False

class DatasetMetaData:
    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(DatasetMetaData.parse_float(four_nps[x*4:x*4+4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_path, annotations,img_width=None,img_height=None, sigma=8.0):
        self.idx = idx
        self.img_path = img_path
        self.img = None
        self.sigma = sigma

        if img_width is None or img_height is None:
            print("#### DatasetMetaData: loading image to get its size: you might want to enter width/height to annotations")
            self.img = cv2.imread(img_path)
            # img_str = open(meta.img_path, 'rb').read()
            # nparr = np.fromstring(img_str, np.uint8)
            # self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.img = None
            img_width=self.img.shape[1]
            img_height=self.img.shape[0]

        self.height = img_height
        self.width = img_width

        self.__num_parts = len(common.BC_parts)
        self.__limb_vecs = common.BC_pairs

        skeletons = []
        for ann in annotations:
            xs = ann[0::3]
            ys = ann[1::3]
            vs = ann[2::3]

            joints = [(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)] 
            joints += [(-1000, -1000)]  # background
            skeletons.append(joints)

        self.skeletons = skeletons

    @jit
    def get_heatmap(self, target_size):
        heatmap = np.zeros((self.__num_parts, self.height, self.width), dtype=np.float32)

        for joints in self.skeletons:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                DatasetMetaData.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    @staticmethod
    @jit(nopython=True)
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    @jit
    def get_vectormap(self, target_size):
        vectormap = np.zeros((len(self.__limb_vecs)*2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((self.__num_parts, self.height, self.width), dtype=np.int16)
        for joints in self.skeletons:
            for plane_idx, (j_idx1, j_idx2) in enumerate(self.__limb_vecs):

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
                    continue

                DatasetMetaData.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p*2+0] /= countmap[p][y][x]
            vectormap[y][x][p*2+1] /= countmap[p][y][x]

        if target_size:
            vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        return vectormap.astype(np.float16)

    @staticmethod
    @jit(nopython=True)
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                perpendicular_dist = abs(bec_x * vec_y - bec_y * vec_x)

                if perpendicular_dist > threshold:
                    continue

                countmap[plane_idx][y][x] += 1

                vectormap[plane_idx*2+0][y][x] = vec_x
                vectormap[plane_idx*2+1][y][x] = vec_y


class BCToolPoseDataReader(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        global mplset
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(BCToolPoseDataReader.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(BCToolPoseDataReader.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = vectmap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        plt.imshow(BCToolPoseDataReader.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(BCToolPoseDataReader.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, anns_file,  is_train=True):
        self.is_train = is_train
        self.anns_file = anns_file
        self.path_to_kps = json.load(open(anns_file))
        logger.info('Found %d anns in file: %s'%(self.size(), anns_file))

    def size(self):
        return len(self.path_to_kps)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass

        keys = list(self.path_to_kps.keys())
        for idx in idxs:
            img_path = keys[idx]
            image_meta = self.path_to_kps[keys[idx]]
            anns = image_meta['keypoint_sets']

            # avoid too much empty images
            # num_annotated_persons = np.sum(1 for kps in anns if not all(val==0 for val in kps))
            num_annotated_persons = len(anns) # assumes no empty annotations exist
            if num_annotated_persons == 0 and random.uniform(0, 1) > 0.2:
                continue
            img_width = image_meta['img_width'] if 'img_width' in image_meta else None
            img_height = image_meta['img_height'] if 'img_height' in image_meta else None
            
            yield [DatasetMetaData(idx, img_path, anns,img_width=img_width,img_height=img_height, sigma=8.0)]


def read_image_url(metas):
    for meta in metas:
        # if meta.img is not None:
        img_str = open(meta.img_path, 'rb').read()
        if not img_str:
            logger.warning('Image not read, path=%s' % meta.img_url)
            raise Exception()
        nparr = np.fromstring(img_str, np.uint8)
        meta.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return metas


def get_dataflow(anns_file, is_train ,augment=True):
    ds = BCToolPoseDataReader(anns_file, is_train=is_train)       # read data from lmdb
    if is_train:
        ds = MapData(ds, read_image_url)
        if augment:
            ds = MapDataComponent(ds, pose_random_scale)
            ds = MapDataComponent(ds, pose_rotation)
            ds = MapDataComponent(ds, pose_flip)
            ds = MapDataComponent(ds, pose_resize_shortestedge_random)
            ds = MapDataComponent(ds, pose_crop_random)
        else:
            ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
            ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        # augs = [
        #     imgaug.RandomApplyAug(imgaug.RandomChooseAug([
        #         imgaug.GaussianBlur(max_size=3)
        #     ]), 0.7)
        # ]
        # ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() * 1)
    else:
        ds = MultiThreadMapData(ds, nr_thread=16, map_func=read_image_url, buffer_size=1000)
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)

    return ds

def get_dataflow_batch(anns_file, is_train, batchsize, augment=True):
    logger.info('Dataflow anns_file=%s' % anns_file)
    ds = get_dataflow(anns_file, is_train , augment=augment)
    ds = BatchData(ds, batchsize)
    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()

# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''

#     from pose_augment import set_network_input_wh, set_network_scale
#     # set_network_input_wh(368, 368)
#     set_network_input_wh(480, 320)
#     set_network_scale(8)

#     # df = get_dataflow('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/')
#     df = _get_dataflow_onlyread('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/')
#     # df = get_dataflow('/root/coco/annotations', False, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')

#     from tensorpack.dataflow.common import TestDataSpeed
#     TestDataSpeed(df).start()
#     sys.exit(0)

#     with tf.Session() as sess:
#         df.reset_state()
#         t1 = time.time()
#         for idx, dp in enumerate(df.get_data()):
#             if idx == 0:
#                 for d in dp:
#                     logger.info('%d dp shape={}'.format(d.shape))
#             print(time.time() - t1)
#             t1 = time.time()
#             BCToolPoseDataReader.display_image(dp[0], dp[1].astype(np.float32), dp[2].astype(np.float32))
#             print(dp[1].shape, dp[2].shape)
#             pass

#     logger.info('done')
