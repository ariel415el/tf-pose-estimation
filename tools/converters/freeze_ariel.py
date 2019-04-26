import argparse
import logging
import os
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
import tensorflow as tf
from tf_pose.networks import get_network, model_wh, _get_base_path
from tensorflow.tools.graph_transforms import TransformGraph
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


#if __name__ == '__main__':
with tf.Graph().as_default() as graph:
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--ckp', type=str, help='checkpoint path')
    parser.add_argument('--trainable',  action='store_true')
    args = parser.parse_args()

    w, h = 432,368 
    input_node = tf.placeholder(tf.float32, shape=(1, h, w, 3), name='image')
    net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, None, trainable=args.trainable)
    
    with tf.Session(config=config) as sess:
        variables_to_restore = slim.get_variables_to_restore()
        #variable_averages = tf.train.ExponentialMovingAverage(0.9997)
        #avg_vars = variable_averages.variables_to_restore()
        loader = tf.train.Saver(variables_to_restore)
        #saver = tf.train.import_meta_graph(args.ckp + '.meta', clear_devices=True)
        import pdb;pdb.set_trace()
        loader.restore(sess, args.ckp)
        
        #tf.train.write_graph(sess.graph_def, os.path.dirname(args.ckp), args.name + '.pb', as_text=True)

        transforms = ['add_default_attributes',
                      'remove_nodes(op=Identity, op=CheckNumerics)',
                      'fold_batch_norms', 'fold_old_batch_norms',
                      'strip_unused_nodes', 'sort_by_execution_order']
        transformed_graph_def = TransformGraph(tf.get_default_graph().as_graph_def(),'image', ["Openpose/concat_stage7"], transforms)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            transformed_graph_def,  # The graph_def is used to retrieve the nodes
            ["Openpose/concat_stage7"]  # The output node names are used to select the useful nodes
        )
        with tf.gfile.GFile(os.path.join(os.path.dirname(args.ckp), "freeze_2.pb"), "wb") as f:
            f.write(output_graph_def.SerializeToString())

