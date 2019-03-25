import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)
import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from common import get_sample_images
from pose_dataset import  CocoToolPoseDataReader
from networks import get_network
import common

def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    val_image = cv2.cvtColor(val_image.astype(np.uint8), cv2.COLOR_BGR2RGB) 
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-like inference')
    parser.add_argument('--out_path', type=str, default='./models/custom_training')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--input-width', type=int, default=656)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--model_name', type=str, default="mobilenet_thin")
    args = parser.parse_args()
    test_images = get_sample_images(args.input_width, args.input_height)
    num_images = len(test_images)
    if not os.path.exists(args.out_path):
         os.makedirs(args.out_path)

    output_h = args.input_height / 8
    output_w = args.input_width / 8
    org_graph = tf.Graph()
    with org_graph.as_default():
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            input_node = tf.placeholder(tf.float32, shape=(num_images, args.input_height, args.input_width, 3), name='my_image')

        outputs = []
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
            with tf.variable_scope(tf.get_variable_scope()):
               net, pretrain_path, last_layer = get_network(args.model_name, input_node)#, trainable=False)
               outputs.append(net.get_output())
        outputs = tf.concat(outputs, axis=0, name='external_concat')

        saver = tf.train.Saver(max_to_keep=100)
        sess  = tf.Session()
        saver.restore(sess, args.checkpoint)
        print(tf.trainable_variables())
    gf = org_graph.as_graph_def()
    import pdb;pdb.set_trace()

    new_graph = tf.Graph()
    with new_graph.as_default():
       input_node_new = tf.placeholder(tf.float32, shape=(num_images, args.input_height, args.input_width, 3), name='my_image_new')
       tf.import_graph_def(gf,input_map={'my_image':input_node_new})
       net_out = new_graph.get_tensor_by_name("import/Openpose/concat_stage7:0")
       sess_new = tf.Session()
       inp =  np.array(test_images)
       tf.global_variables_initializer().run(session=sess_new)
       tf.tables_initializer().run(session=sess_new)
   
       outputMat = sess_new.run(net_out, feed_dict={input_node_new: inp})
       pafMat, heatMat = outputMat[:, :, :,len(common.BC_parts):], outputMat[:, :, :, :len(common.BC_parts)]
       for i in range(num_images):
            test_result = CocoToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i], as_numpy=True).astype(float)
            test_result = cv2.resize(test_result, (640, 640))
            test_result = test_result.reshape([640, 640, 3]).astype(float)
            # cv2.imwrite(os.path.join(args.out_path,"all_hm_%d.png"%i),heatMat[i][-1])
            cv2.imwrite(os.path.join(args.out_path,"pic_%d.png"%i),test_result)
