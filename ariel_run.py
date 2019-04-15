import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
import json
import os

def translate_op_top_coco(kps):
        coco_keypoints_indices = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10] # taken from openpose master code  (filesustem cpp files)
        assert( len(kps) == 54 )
        result = []
        for i in coco_keypoints_indices:
              result +=kps[3*i:3*i+3]
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--images', type=str)
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--in_name', type=str, default="image:0")
    parser.add_argument('--out_name', type=str, default="Openpose/concat_stage7:0")

    args = parser.parse_args()
    graph_path = get_graph_path(args.model) if args.model in ["cmu", ' mobilenet_thin'] else args.model
    model_name = args.model if args.model in ["cmu", ' mobilenet_thin'] else os.path.splitext(os.path.basename(graph_path))[0]
    out_dir = args.images + "_out_" + model_name +"_"+  args.resize
    if not os.path.exists(out_dir): 
       os.makedirs(out_dir)
    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(graph_path, input_name=args.in_name, output_name=args.out_name, target_size=(432, 368))
    else:
        print("### graph path = ",graph_path)
        e = TfPoseEstimator(graph_path, input_name=args.in_name, output_name=args.out_name, target_size=(w, h))
    total_time = 0
    coco_json = []
    image_id=0
    num_person =0
    # estimate human poses from a single imaige !
    for fname in os.listdir(args.images):
        image_id += 1
        fpath = os.path.join(args.images, fname)
        image = common.read_imgfile(fpath, w, h)
        print("###",fname)
        image_h, image_w = image.shape[:2]
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            continue
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        total_time += elapsed
        logger.info('inference image: %s in %.4f seconds.' % (fname, elapsed))
        for human in humans:
            num_person += 1
            kps= []
            p = {}
            for i in range(18):
                if i in human.body_parts.keys():
                    kps += [ human.body_parts[i].x * image_w,  human.body_parts[i].y * image_h, human.body_parts[i].score]
                else:
                    kps += [0,0,0]
            p["filename"]=fname
            p['category_id'] = 1
            p['image_id'] = image_id
            p['keypoints'] = translate_op_top_coco(kps)
            coco_json += [p]
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        #cv2.imwrite(os.path.join(out_dir,fname), bgimg)
        cv2.imwrite(os.path.join(out_dir,fname), image)
        print("saving image to: ",os.path.join(out_dir,fname))
    if image_id > 0 and num_person > 0 : 
        logger.info('time per image: %.4f seconds.' % (total_time/float(image_id)))
        logger.info('time per person: %.4f seconds.' % (total_time/float(num_person)))
    json_path =  os.path.join(os.path.dirname(args.images), "tf-openpose_%s_%s.json"%(model_name, args.resize))
    print("saving json to: ",json_path)
    json.dump(coco_json, open(json_path, 'w'))
