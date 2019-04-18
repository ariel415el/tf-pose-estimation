import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.pyplot import cm
import numpy as np
### Written by Ariel-e

def showKpsAnns(anns, vis_thresh=0.0):
    if len(anns) == 0:
        return 0
    if 'keypoints' not in anns[0]:
        raise Exception('datasetType not supported')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    # polygons = []
    colors = cm.rainbow(np.linspace(0,1,len(anns)))
    for i,ann in enumerate(anns):
        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            sks = np.array([[15, 13]
             ,[13,11]
            ,[16,14]
            ,[14,12]
            ,[11,12]
            ,[5,11]
            ,[6,12]
            ,[5,6]
            ,[5,7]
            ,[6, 8]
            ,[7,9]
            ,[8,10]
            ,[1,2]
            ,[0,  1]
            ,[0,2]
            ,[1,  3]
            ,[2,4]
            ,[3,  5]
            ,[4,6]])
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]
            for sk in sks:
                if np.all(v[sk] > vis_thresh):
                    plt.plot(x[sk],y[sk], linewidth=1, color=colors[i])
            plt.plot(x[v > vis_thresh], y[v > vis_thresh],'o',markersize=1, markerfacecolor=colors[i], markeredgecolor='k',markeredgewidth=1)
            # plt.plot(x[v>1], y[v>1],'o',markersize=1, markerfacecolor=colors[i], markeredgecolor=colors[i], markeredgewidth=1)
        # p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        # p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
        # ax.add_collection(p)

if __name__ == '__main__':
    imgs_dir = sys.argv[1]
    print( imgs_dir)
    kps_file = sys.argv[2]
    d = json.load(open(kps_file))
    # if 'images' in d.keys():
    if isinstance(d, (dict,)):
        d = d['annotations']
    out_dir = os.path.splitext(kps_file)[0] + "_debug_imgs"
    if not os.path.exists(out_dir):
        print( "creating folder : " +  out_dir)
        os.makedirs(out_dir)
    done_ims= []
    num_errors = 0
    for ann in d:
        if "filename" in ann.keys():
            filename = ann["filename"]
            anns_of_same_image = [x for x in d if filename ==  x["filename"]]
        elif "file_name" in ann.keys():
            filename = os.path.basename(ann["file_name"])
            ann["file_name"] = filename
            anns_of_same_image = [x for x in d if filename ==  x["file_name"]]
        elif "fname_no_ext" in ann.keys():
            fname_no_ext = ann["fname_no_ext"]
            mathing_names = [x for x in os.listdir(imgs_dir) if os.path.splitext(x)[0] == fname_no_ext]
            if(len(mathing_names) != 1):
                num_errors  += 1
            filename = mathing_names[0]
            anns_of_same_image = [x for x in d if fname_no_ext ==  x["fname_no_ext"]]

        else:
            # image_id = ann["image_id"]
            image_id = str(ann["image_id"])
            mathing_names = [x for x in os.listdir(imgs_dir) if image_id in x ]
            if(len(mathing_names) != 1):
                num_errors  += 1
            filename = mathing_names[0]
            anns_of_same_image = [x for x in d if image_id ==  x["image_id"]]



        if filename in done_ims:
            continue
        done_ims += [filename]

        print( "working on image: ",filename)
        file_path = os.path.join(imgs_dir, filename)
                # show image
        I = io.imread(file_path)
        plt.imshow(I,interpolation='none')
        plt.axis('off')
        ax = plt.gca()
        showKpsAnns(anns_of_same_image, 0.0)
        plt.savefig(os.path.join(out_dir,filename),dpi=200, bbox_inches='tight')
        plt.clf()
    print( "errors in" , num_errors  ," images")
