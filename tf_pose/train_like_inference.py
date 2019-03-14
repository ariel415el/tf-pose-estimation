import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)
import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from common import get_sample_images
from pose_dataset import  CocoPose
from networks import get_network


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
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(num_images, args.input_height, args.input_width, 3), name='image')

    outputs = []
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope()):
            net, pretrain_path, last_layer = get_network(args.model_name, input_node)#, trainable=False)
            outputs.append(net.get_output())
    outputs = tf.concat(outputs, axis=0)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint)

        #import pdb;pdb.set_trace()i
        inp =  np.array(test_images)
        print(inp.shape)
        outputMat = sess.run(outputs, feed_dict={input_node: inp})
        pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]
        for i in range(num_images):
            test_result = CocoPose.display_image(test_images[i], heatMat[i], pafMat[i], as_numpy=True).astype(float)
            test_result = cv2.resize(test_result, (640, 640))
            test_result = test_result.reshape([640, 640, 3]).astype(float)
            # cv2.imwrite(os.path.join(args.out_path,"all_hm_%d.png"%i),heatMat[i][-1])
            cv2.imwrite(os.path.join(args.out_path,"pic_%d.png"%i),test_result)
