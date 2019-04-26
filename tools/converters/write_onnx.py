import sys
import math
import os
import numpy as np
import tensorflow as tf

ckp = sys.argv[1]

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(ckp + ".meta"))
    saver.restore(sess, os.path.join(ckp))

    with open(os.path.join(os.path.dirname(ckp),"freeze_onnx.onnx", "wb")) as f:
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        const_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph_def,  #transformed_graph_def,  # The graph_def is used to retrieve the nodes
            ["Openpose/concat_stage7"]  # The output node names are used to select the useful nodes
        )
        tf.import_graph_def(const_graph_def)
        graph = tf.Graph().as_default()
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, input_names=["image:0"], output_names=["Openpose/concat_stage7:0"])
        model_proto = onnx_graph.make_model("test")
        f.write(model_proto.SerializeToString())
