import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import os

with tf.Session() as sess:
    # model_filename ='/home/host_tf-pose/models/graph/mobilenet_thin/graph_opt.pb'
    model_filename = sys.argv[1]
    with tf.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        if len(sys.argv) > 2:
             g_in = tf.import_graph_def(graph_def,name=sys.argv[2])
        else:
             g_in = tf.import_graph_def(graph_def)
LOGDIR=os.path.dirname(model_filename)
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
