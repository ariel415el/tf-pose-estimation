import argparse
import logging

import tensorflow as tf
from tf_pose.networks import get_network
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet / mobilenet_thin')
    parser.add_argument('--ckp', type=str, help='checkpoint path')
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--width', type=int, default=656)
    parser.add_argument('--height', type=int, default=368)
    parser.add_argument('--trainable',  action='store_true')

    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1,args.height, args.width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess, trainable=args.trainable, ckp=args.ckp)

        tf.train.write_graph(sess.graph_def, os.path.dirname(args.ckp), args.name + '.pb', as_text=True)

        saver = tf.train.Saver(max_to_keep=100)
        saver.save(sess, os.path.join(os.path.dirname(args.ckp), "generated_checkpoint"), global_step=1)
