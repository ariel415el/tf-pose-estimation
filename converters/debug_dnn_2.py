import tensorflow as tf
import sys
from tf_pose.networks import get_network
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.tools.graph_transforms import TransformGraph
import os
import tf_pose.common
import matplotlib.pyplot as plt
import cv2
from tf_pose.pose_augment import *
from tf_pose.common import *
from tf_pose.pose_dataset import  CocoToolPoseDataReader
import onnx
from onnx import numpy_helper
import matplotlib
matplotlib.use('Agg')

def get_onnx_dict(onnx_path, layer_name=None):
    data_dict = {}
    model = onnx.load(onnx_path)
    layers = model.graph.initializer
    for layer in layers:
        if layer_name in layer.name:
            print("Onnx lyaer: ", layer.name)
            w = numpy_helper.to_array(layer)
            data_dict[layer.name.split(":")[0]] = w,"NA"
    return data_dict


def extract_heat_maps_from_ckp(ckp,image_path, trainable=False):
    test_images = get_sample_images(432, 368)[:1]
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(len(test_images), 368, 432, 3), name='image')

    outputs = []
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope()):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, trainable=trainable)
            outputs.append(net.get_output())
    outputs = tf.concat(outputs, axis=0)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckp)
        #
        # outputMat = sess.run(outputs, feed_dict={input_node: np.array(test_images)})
        # pafMat, heatMat = outputMat[:, :, :, 15:], outputMat[:, :, :, :15]

        # unreported = sess.run(tf.report_uninitialized_variables())
        # tensor_image = graph.get_tensor_by_name("image:0")
        # tensor_output = graph.get_tensor_by_name("Openpose/concat_stage7:0")
        # tensor_heatMat = tensor_output[:, :, :, :15]
        # tensor_pafMat = tensor_output[:, :, :, 15:]
        # hm,pm = sess.run([tensor_heatMat,tensor_pafMat],feed_dict={tensor_image:[image]})

        # tmp2 = pm[0].transpose((2, 0, 1))
        # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
        # plt.imshow(tmp2_odd)

        inp =  np.array(test_images)
        outputMat = sess.run(outputs, feed_dict={input_node: inp})
        pafMat, heatMat = outputMat[:, :, :, 15:], outputMat[:, :, :, :15]
        for i in range(len(test_images)):
            test_result = CocoToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i], as_numpy=True).astype(float)
            test_result = cv2.resize(test_result, (640, 640))
            test_result = test_result.reshape([640, 640, 3]).astype(float)
            # cv2.imwrite(os.path.join(args.out_path,"all_hm_%d.png"%i),heatMat[i][-1])
            cv2.imwrite(os.path.join(os.path.dirname(ckp),"ckp_test_%d.png"%i), test_result)
        # exit()

def extract_heat_maps_from_pb(pb_file, image_path):
    test_images = get_sample_images(432, 368)[:1]
    # with tf.device(tf.DeviceSpec(device_type="CPU")):
    #     input_node = tf.placeholder(tf.float32, shape=(len(test_images), 368, 432, 3), name='my_image')
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')#,input_map={'image':input_node})
            with tf.Session() as sess:
                # unreported = sess.run(tf.report_uninitialized_variables())
                # tensor_image = graph.get_tensor_by_name("image:0")
                # tensor_output = graph.get_tensor_by_name("Openpose/concat_stage7:0")
                # tensor_heatMat = tensor_output[:, :, :, :15]
                # tensor_pafMat = tensor_output[:, :, :, 15:]
                # hm,pm = sess.run([tensor_heatMat,tensor_pafMat],feed_dict={tensor_image:[image]})
                # tmp2 = pm[0].transpose((2, 0, 1))
                # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
                # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
                # plt.imshow(tmp2_odd)

                inp = np.array(test_images)
                outputs = graph.get_tensor_by_name("Openpose/concat_stage7:0")
                outputMat = sess.run(outputs, feed_dict={"image:0": inp})
                pafMat, heatMat = outputMat[:, :, :, 15:], outputMat[:, :, :, :15]
                for i in range(len(test_images)):
                    test_result = CocoToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i],
                                                                       as_numpy=True).astype(float)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    # cv2.imwrite(os.path.join(args.out_path,"all_hm_%d.png"%i),heatMat[i][-1])
                    cv2.imwrite(os.path.join(os.path.dirname(pb_file), "pb_test_%d.png" % i), test_result)
                return

def save_pb_file(ckp, path,trainable=False,do_transforms=False):
    with tf.Graph().as_default() as graph:
        input_node = tf.placeholder(tf.float32, shape=(None, 368, 432, 3), name='image')
        with tf.device(tf.DeviceSpec(device_type="GPU")):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, None, trainable=trainable)
        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            loader = tf.train.Saver(variables_to_restore)
            loader.restore(sess, ckp)
            graph_def = tf.get_default_graph().as_graph_def()
            if do_transforms:
                transforms = ['add_default_attributes',
                              'remove_nodes(op=Identity, op=CheckNumerics)',
                              'strip_unused_nodes', 'sort_by_execution_order',
                              'fold_batch_norms', 'fold_old_batch_norms']
                graph_def = TransformGraph(graph_def, 'image',["Openpose/concat_stage7"], transforms)

            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                graph_def,  # The graph_def is used to retrieve the nodes
                ["Openpose/concat_stage7"]  # The output node names are used to select the useful nodes
            )
            with tf.gfile.GFile(path, "wb") as f:
                f.write(graph_def.SerializeToString())

def get_ops_from_pb(pb_file, layer_name):
    name_to_var_and_type = {}
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            with tf.Session() as sess:
                ops = tf.get_default_graph().get_operations()
                if layer_name is not None:
                    ops = [op for op in ops if layer_name in op.name]
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    for op in ops:
                        try:
                            val = op.values()[0].eval()
                            name_to_var_and_type[op.name] = (val, op.type)
                        except Exception as e:
                            pass
                return name_to_var_and_type

def get_ops_from_ckp(ckp, layer_name=None):
    name_to_var_and_type = {}
    with tf.Graph().as_default() as graph:
        input_node = tf.placeholder(tf.float32, shape=(1, 368, 432, 3), name='image')
        with tf.device(tf.DeviceSpec(device_type="GPU")):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, None, trainable=False)
        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            loader = tf.train.Saver(variables_to_restore)
            loader.restore(sess, ckp)
            ops = tf.get_default_graph().get_operations()
            if layer_name is not None:
                ops = [op for op in ops if layer_name in op.name]
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                for op in ops:
                    try:
                        val = op.values()[0].eval()
                        name_to_var_and_type[op.name] = (val, op.type)
                    except Exception as e:
                        pass
            return name_to_var_and_type

def compare_data_dicts(dict_source, dict_ref):
    missing_layers = 0
    transposed_layers = 0
    different_layers = 0
    missed_layers = []
    for op_name in dict_source:
        source_val = dict_source[op_name][0]
        source_type = dict_source[op_name][1]

        if op_name not in dict_ref:
            missing_layers += 1
            print("Op: %s of type %s is missing from ref model" % (op_name, source_type))
            missed_layers += [op_name]
            continue
        ref_val = dict_ref[op_name][0]
        ref_type = dict_ref[op_name][1]

        layers_same = np.sum(ref_val == source_val) == source_val.size
        layers_transposed = np.sum(ref_val == source_val.transpose()) == source_val.size

        if layers_same:
            print("Op OK: ", op_name)
        elif layers_transposed:
            transposed_layers+=1
            print("Op transposed: ", op_name)
        else:
            print("Op: different", op_name)
            print("\tRef type "+ref_type)
            print("\tSource type "+ source_type)
            print("\tShapes: " + source_val.shape,ref_val.shape)
            print("\tDiff: source has %d/%d of pb_dict_ref"%(np.sum(source_val == ref_val), source_val.size))
            different_layers += 1

    print("missing_layers: ", missing_layers)
    print("different_layers: ", different_layers)
    print("transposedlayers: ", transposed_layers)

def create_onnx_from_pb(pb_path, onnx_path):
    os.system(" python3 -m tf2onnx.convert --input %s --inputs image:0 --outputs Openpose/concat_stage7:0 --verbose --output  %s"%(pb_path, onnx_path))


def main():
    TRAINABLE=True
    OPTIMIZE=False
    ckp = sys.argv[1]
    pb_out_path = os.path.join(os.path.dirname(ckp), "freeze_ariel.pb")
    image_path = "/home/briefcam/Projects/ArielE/tf-pose-git/images/vilage.jpg"
    # ref_model_path = "/home/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_opt_constant.pb"
    onnx_model_path = os.path.join(os.path.dirname(ckp), "freeze_ariel.onnx")
    layer_name = "Openpose/MConv_Stage1_L1_1"

    save_pb_file(ckp, pb_out_path, trainable=TRAINABLE, do_transforms=OPTIMIZE)
    create_onnx_from_pb(pb_out_path, onnx_model_path)

    extract_heat_maps_from_pb(pb_out_path, image_path)
    exit()
    onnx_dict = get_onnx_dict(onnx_model_path, layer_name)
    # # extract_heat_maps_from_ckp(ckp, image_path)

    # ckp_dict = get_ops_from_ckp(ckp, layer_name)
    # script_dict = get_ops_from_pb(script_model_path,layer_name)
    saved_dict = get_ops_from_pb(pb_out_path,layer_name)
    # ref_dict = get_ops_from_pb(ref_model_path,layer_name)
    print("Comparing")
    compare_data_dicts(onnx_dict, saved_dict)
    print("Done")

if __name__ == '__main__':
    main()
