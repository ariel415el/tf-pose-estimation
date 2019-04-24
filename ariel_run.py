import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import json
import os
from tf_pose.postProcess.python_paf_process import NUM_PART, NUM_HEATMAP
from tf_pose.common import draw_humans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--images', type=str)
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    parser.add_argument('--upsample_heatmaps', type=int, default=4)
    parser.add_argument('--in_name', type=str, default="image:0")
    parser.add_argument('--out_name', type=str, default="Openpose/concat_stage7:0")

    parser.add_argument('--debug_images', action='store_true')

    args = parser.parse_args()
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = args.images + "_out_" + model_name
    if not os.path.exists(out_dir): 
       os.makedirs(out_dir)
    w, h = model_wh(args.resize)
    tf_estimator = TfPoseEstimator(args.model, input_name=args.in_name, output_name=args.out_name, input_size=(w, h))
    total_time = 0
    coco_json = {}
    # estimate human poses from a single imaige !
    for fname in os.listdir(args.images):
        print("Working on: ",fname)
        
        fpath = os.path.join(args.images, fname)
        org_image = cv2.imread(fpath, cv2.IMREAD_COLOR)
        org_image_h, org_image_w = org_image.shape[:2]
        coco_json[os.path.abspath(fpath)] = {"keypoint_sets" : [], "img_width": org_image_w, "img_height" : org_image_h}
        if org_image is None:
            print('\tImage can not be read, path=%s' % args.image)
            continue
        humans = tf_estimator.inference(org_image, upsampleHeatMaps=args.upsample_heatmaps)
        coco_json[os.path.abspath(fpath)]["keypoint_sets"] = humans
        # import pdb;pdb.set_trace()

        if args.debug_images:
            debug_image = draw_humans(org_image, np.array(humans), imgcopy=False)
            print("\tSaving image to: ",os.path.join(out_dir,fname))
            cv2.imwrite(os.path.join(out_dir, fname), debug_image)

    json_path =  os.path.join(os.path.dirname(args.images), "tf-openpose_%s.json"%model_name)

    print("###########################")
    print("Saving json to: ",json_path)
    json.dump(coco_json, open(json_path, 'w'))
