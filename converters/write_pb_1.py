import sys
import math
import os
import numpy as np
import tensorflow as tf

ckp = sys.argv[1]

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(ckp + ".meta"))
    saver.restore(
        sess, os.path.join(ckp))

    with open(os.path.join(os.path.dirname(ckp),"freeze_1.pb", "wb")) as f:
        graph = tf.get_default_graph().as_graph_def(add_shapes=True)
        f.write(graph.SerializeToString())
