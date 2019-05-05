import skimage.io as io
import sys, os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import argparse
import time
import cv2

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tf_pose"))
sys.path.insert(0,parent_dir)
from debug_tools import draw_humans


## Taken from pycocotools
## Compute OKS betaween each Det to each gt skeketon
def computeOks(gt_kps, det_kps):
    # dimention here should be Nxm
    if len(gt_kps) == 0 or len(det_kps) == 0:
        return np.array([])

    ious = np.zeros((len(gt_kps), len(det_kps)))
    sigmas = np.array([2.0, 2.0, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gt_kps):
        # create bounds for ignore regions(double the gt bbox)
        xg = gt[0::3]
        yg = gt[1::3]
        vg = gt[2::3]
        bb_w = max(xg) - min(xg)
        bb_h = max(yg) - min(yg)
        for i, dt in enumerate(det_kps):
            xd = dt[0::3]
            yd = dt[1::3]
            vd = dt[2::3]
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
            size_factor = bb_w*bb_h/2
            e = (dx**2 + dy**2) / (vars * size_factor * 2 + np.spacing(1))
            normalize_by = e.shape[0]
            e=e[(vg > 0) & (vd > 0)]
            if e.shape[0] > 0:
                exp = np.exp(-e)
                ious[j, i] = np.sum(exp) / normalize_by

    return ious

# Returns a  version of the input annotations filtered by the given thrsholds
def filter_anns(anns, vis_th, min_kps, min_height, ignore_head=False):
    new_anns = []
    invalid_anns = []
    for ann in anns:
        ann_copy = np.array(ann, copy=True)
        if ignore_head:
            ann_copy[2] = 0 # drop visibility of first kp, middle head
        ann_copy[2::3][ann_copy[2::3] < vis_th] = 0
        y_vals = ann_copy[1::3]
        bbox_height = max(y_vals) - min(y_vals)
        num_valid_kps = np.count_nonzero(ann_copy[2::3])
        if (num_valid_kps >= min_kps) and (bbox_height > min_height):
            new_anns += [ann_copy]
        else:
            invalid_anns += [ann] # load anns as they are
    return np.array(new_anns), np.array(invalid_anns)

## Compute recal and precision values for the given thresholds
def evaluate(gt_kps, det_kps, iou_threshold, debug_gt_vis_th, get_debug_image=False):
    fps = 0
    tps = 0
    gts = 0
    dts = 0
    t = time.time()
    # Create debug foldr
    # For each Gt compute similarity to all detections
    debug_images = []
    for i, image_path in enumerate(gt_kps):
        gt = gt_kps[image_path]
        if len(gt) == 0:
            continue
        gts += len(gt)
        if image_path in det_kps:
            dt = det_kps[image_path]
            if len(dt) == 0:
                continue
            dts += len(dt)  

            ious = computeOks(gt, dt) # this is a (gt)x(det) matrices of ious
            ious = ious.transpose()

            first_matches_idxs = np.argmax(ious, axis=1) # best det for each gt
            first_matches_vals = ious[np.arange(ious.shape[0]), first_matches_idxs]
            valid_matches_indices = np.where(first_matches_vals > iou_threshold)[0] # best det for each gt
            tps += len(valid_matches_indices)
            debug_image = None
            if get_debug_image:
                im = cv2.imread(image_path)# cv2.cvtColor(, cv2.COLOR_BGR2RGB)
                draw_humans(im, gt,color=[0,255,0])
                draw_humans(im, dt,color=[255,0,0])
                cv2.putText(im,"GTs: %d"%len(gt), (10,30), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)    
                cv2.putText(im,"TPs: %d"%len(valid_matches_indices), (10,60), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0), 2)
                for k in range(len(first_matches_idxs)):
                    vis = np.where(gt[first_matches_idxs[k]][2::3] >= debug_gt_vis_th)[0]
                    if len(vis) > 0:
                        first_visible_idx = vis[0]
                        gt_x = gt[first_matches_idxs[k]][3*first_visible_idx]
                        gt_y = gt[first_matches_idxs[k]][3*first_visible_idx + 1]
                        cv2.putText(im,"%.2f"%first_matches_vals[k], (int(gt_x) -10 ,int(gt_y) - 20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255) ,1)    
                debug_images += [im]

    # Compute recall precision
    if gts == 0:
        recall = 1 
    else:
        recall = min(1,max(0,tps / float(gts)))
    if dts == 0 :
        precision = 1
    else:
        precision = min(1,max(0,tps / float(dts))) # tps may be larger than dets
    print("Evaluation of %d dts and %d gts took %f sec"%(dts,gts,time.time()-t))
    return recall, precision, debug_images

def compute_roc(gt_anns, det_anns, iou_th, gt_vis_th, min_kps, min_height, ignore_head, debug_vis_th):
    # Clean invalid GTs
    gt_invalid_anns = {}
    for im_name in gt_anns:
        gt_anns[im_name], gt_invalid_anns[im_name] = filter_anns(gt_anns[im_name],
                                                                    vis_th=gt_vis_th,
                                                                    min_kps=min_kps,
                                                                    min_height=min_height,
                                                                    ignore_head=ignore_head)
    
    # Iterate over vis threshoulds and compute R/P
    recall_vals=[] 
    precision_vals=[]
    debug_images_set=[]
    visThrs = np.arange(0, 1.05, 0.05)
    for i,th in enumerate(visThrs):
        debug = (abs(th-debug_vis_th) < 0.01 )
        det_invalid_anns = {}
        detections_copy = det_anns.copy()
        for im_name in detections_copy:
            detections_copy[im_name], det_invalid_anns[im_name] = filter_anns(detections_copy[im_name],
                                                                                vis_th=th,
                                                                                min_kps=min_kps,
                                                                                min_height=min_height,
                                                                                ignore_head=ignore_head)
        recall, precision, images_set = evaluate(gt_anns,
                                detections_copy,
                                iou_th,
                                debug_gt_vis_th=gt_vis_th,
                                get_debug_image=debug)

        if debug:
            debug_images_set = images_set
        recall_vals += [recall]
        precision_vals += [precision]
        print("%d images done :%d/%d ths done"%(len(detections_copy), i, len(visThrs)))

    return recall_vals, precision_vals, debug_images_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', required=True)
    parser.add_argument('--det_file', required=True)
    parser.add_argument('--iou_th',type=float, default=0.5)
    parser.add_argument('--gt_fixed_vis_th',type=float, default=1) # relevant if gt has varying visibilities (Alphapose) or vis is in other range (coco { 0,1,2 })
    parser.add_argument('--min_kps',type=int, default=3) 
    parser.add_argument('--min_bbox_height',type=int, default=100) # TODO : use height ratio instaed of absolute
    parser.add_argument('--debug_vis_th',type=float, default=0.2)
    parser.add_argument('--ignore_head',action='store_true')
    args = parser.parse_args()

    # load annotations
    gt_anns=json.load(open(args.gt_file))
    det_anns=json.load(open(args.det_file))
    gt_anns={path:  gt_anns[path]['keypoint_sets'] for path in gt_anns}
    det_anns = {path: det_anns[path]['keypoint_sets'] for path in det_anns}

    recall_vals, precision_vals, debug_images_sets = compute_roc(gt_anns, det_anns,
                iou_th=args.iou_th,
                gt_vis_th=args.gt_fixed_vis_th,
                min_kps=args.min_kps,
                min_height=args.min_bbox_height,
                ignore_head=args.ignore_head,
                debug_vis_th=args.debug_vis_th)

    # print("vis Th %f with iou threshold %f gives recall %f and precision %f"%(visThrs[3], iou_th, recall_vals[3],precision_vals[3]))
    # with open(os.path.join(os.path.dirname(args.det_file), os.path.basename(os.path.splitext(.det_file)[0])+"_res.txt"), "w") as f :
    #     f.write("vis Th %f with iou threshold %f gives recall %f and precision %f"%(visThrs[3], args.iou_th, recall_vals[3],precision_vals[3]))
    #     f.write("\n")

    plt.title('Receiver Operating Characteristic')
    plt.plot(recall_vals, precision_vals, 'b')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(os.path.dirname(args.det_file), "ROC_" + os.path.basename(os.path.splitext(args.det_file)[0])+".png"))

    out_dir = os.path.join(os.path.splitext(args.det_file)[0] + "_debug-th_%f" % args.debug_vis_th)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i,images_set in enumerate(debug_images_sets):
        for j,im in enumerate(images_set):
            cv2.imwrite(os.path.join(out_dir, "im_%d_%d.png"%(i,j)), im)