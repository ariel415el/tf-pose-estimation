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
from tf_pose.pose_dataset import  BCToolPoseDataReader
import onnx
from onnx import numpy_helper
import matplotlib
matplotlib.use('Agg')

def get_onnx_dict(onnx_path, layer_name=None):
    data_dict = {}
    model = onnx.load(onnx_path)
    layers = model.graph.initializer
    for layer in layers:
        print("Onnx lyaer: ", layer.name)
        if layer_name in layer.name:
            w = numpy_helper.to_array(layer)
            data_dict[layer.name.split(":")[0]] = w,"NA"
        else:
            print(layer_name ," not in ", layer.name)
    return data_dict


def extract_heat_maps_from_ckp(ckp,image_path, trainable=False):
    # test_images = get_sample_images(432, 368)[:1]
    test_images = [read_imgfile(image_path,432, 368)]
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

        inp =  np.array(test_images)
        outputMat = sess.run(outputs, feed_dict={input_node: inp})
        pafMat, heatMat = outputMat[:, :, :, 15:], outputMat[:, :, :, :15]
        for i in range(len(test_images)):
            test_result = BCToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i], as_numpy=True).astype(float)
            test_result = cv2.resize(test_result, (640, 640))
            test_result = test_result.reshape([640, 640, 3]).astype(float)
            cv2.imwrite(os.path.join(os.path.dirname(ckp),"ckp_test_%d.png"%i), test_result)

def extract_heat_maps_from_pb(pb_file, image_path):
    # test_images = get_sample_images(432, 368)[:1]
    test_images = [read_imgfile(image_path, 432, 368)]
    # with tf.device(tf.DeviceSpec(device_type="CPU")):
    #     input_node = tf.placeholder(tf.float32, shape=(len(test_images), 368, 432, 3), name='my_image')
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')#,input_map={'image':input_node})
            with tf.Session() as sess:
                inp = np.array(test_images)
                outputs = graph.get_tensor_by_name("Openpose/concat_stage7:0")
                outputMat = sess.run(outputs, feed_dict={"image:0": inp})
                pafMat, heatMat = outputMat[:, :, :, 15:], outputMat[:, :, :, :15]
                for i in range(len(test_images)):
                    test_result = BCToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i],
                                                                       as_numpy=True).astype(float)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    # cv2.imwrite(os.path.join(args.out_path,"all_hm_%d.png"%i),heatMat[i][-1])
                    cv2.imwrite(os.path.join(os.path.dirname(pb_file), "pb_test_%d.png" % i), test_result)
                    cv2.imwrite(os.path.join(os.path.dirname(pb_file), "hm_test_%d.png" % i), heatMat[i][:,:,14]*255)
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
                transforms = [
                    'strip_unused_nodes(type=float, shape="1,368,432,3")',
                    # 'fold_constants(ignoreError=False)',
                    # 'add_default_attributes',
                    'remove_nodes(op=Identity, op=CheckNumerics)',
                    # 'strip_unused_nodes', 'sort_by_execution_order',
                     'fold_batch_norms', 'fold_old_batch_norms'
                    ]
                graph_def = TransformGraph(graph_def, 'image', ["Openpose/concat_stage7"], transforms)

            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                ["Openpose/concat_stage7"]  # The output node names are used to select the useful nodes
            )
            tf.import_graph_def(graph_def, name='')
            with tf.gfile.GFile(path, "wb") as f:
                f.write(tf.get_default_graph().as_graph_def().SerializeToString())

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

def get_ops_from_ckp(ckp, layer_name=None, trainable=False):
    name_to_var_and_type = {}
    with tf.Graph().as_default() as graph:
        input_node = tf.placeholder(tf.float32, shape=(1, 368, 432, 3), name='image')
        with tf.device(tf.DeviceSpec(device_type="GPU")):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, None, trainable=trainable)
        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            loader = tf.train.Saver(variables_to_restore)
            loader.restore(sess, ckp)
            ops = tf.get_default_graph().get_operations()
            if layer_name is not None:
                ops = [op for op in ops if layer_name in op.name]

                for op in ops:
                    try:
                        val = op.values()[0].eval()
                        name_to_var_and_type[op.name] = (val, op.type)
                    except Exception as e:
                        pass
            return name_to_var_and_type

def compare_data_dicts(dict_source, dict_ref):
    transposed_layers = 0
    different_layers = []
    same_layers = []
    missed_layers = []
    for op_name in dict_source:
        source_val = dict_source[op_name][0]
        source_type = dict_source[op_name][1]

        if op_name not in dict_ref:
            print("Op: %s of type %s is missing from ref model" % (op_name, source_type))
            missed_layers += [op_name]
            continue
        ref_val = dict_ref[op_name][0]
        ref_type = dict_ref[op_name][1]

        layers_same = np.sum(ref_val == source_val) == source_val.size
        layers_transposed = np.sum(ref_val == source_val.transpose()) == source_val.size

        if layers_same:
            same_layers += [op_name]
            print("OK: ", op_name)
        elif layers_transposed:
            transposed_layers+=1
            print("Transposed: ", op_name)
        else:
            different_layers += [op_name]
            print("Different", op_name)
            print("\tRef type "+ref_type)
            print("\tSource type "+ source_type)
            print("\tShapes: " , source_val.shape,ref_val.shape)
            print("\tDiff: source has %d/%d of pb_dict_ref"%(np.sum(source_val == ref_val), source_val.size))

    print("missing_layers: ", len(missed_layers))
    print("different_layers: ", len(different_layers))
    print("transposed layers: ", transposed_layers)
    print("same_layers: ", len(same_layers))
    return missed_layers, different_layers, same_layers

def compare_dict_by_values(dict_source, dict_ref):
    pairs = {}
    zzz = [z for z in dict_ref if "Openpose/MConv_Stage1" in z or "Const" in z]
    for key in [z for z in dict_source if "Openpose/MConv_Stage1" in z or "Const" in z]:
        relevant = []
        best_score = 99999
        best_key = None
        print("Source: ",key)
        for key_2 in zzz:
            cur_score = 99999
            q = dict_source[key][0]
            ref = dict_ref[key_2][0]
            if q.shape == ref.shape:
                cur_score = np.mean(np.abs(q - ref))
            elif q.shape == ref.transpose().shape:
                cur_score = np.mean(np.abs(q - ref.transpose()))
            elif len(ref.shape) > 3 and q.shape == ref.transpose(2, 3, 0, 1).shape:
                cur_score = np.mean(np.abs(q - ref.transpose(2, 3, 0, 1)))
            if cur_score < best_score:
                best_key = key_2
                best_score = cur_score
            # same = np.sum(dict_source[key][0] == dict_ref[key_2][0]) > 0.9*dict_source[key][0].size
            # trans = np.sum(dict_source[key][0] == dict_ref[key_2][0].transpose()) > 0.9*dict_source[key][0].size
        if best_score < 0.1:
            relevant += [(best_key, best_score)]
        #     print("\t" + "equal to ", key_2)
        #     relevant += [(key_2, cur_score)]
        pairs[key] = relevant
    return pairs



def create_onnx_from_pb(pb_path, onnx_path):
    # os.system(" python3 -m tf2onnx.convert --input %s --inputs image:0 --outputs Openpose/concat_stage7:0 --verbose --output  %s"%(pb_path, onnx_path))
    os.system(" python3 -m tf2onnx.convert --input %s --inputs image:0 --outputs ace/strided_slice_6:0,ace/strided_slice_7:0 --verbose --output  %s" % (pb_path, onnx_path))

def run_freeze_script(ckp, pb_path):
    graph_def_name=os.path.splitext(os.path.basename(pb_path))[0]
    # os.system("python3  ../run_checkpoint.py --model  --ckp %s --name %s  --resize 432x368"%(ckp, graph_def_name))
    # os.system("python3 -m tensorflow.python.tools.freeze_grap --input_graph=%s --output_graph %s --input_checkpoint %s --output_node_names %s"%(os.path))
    # os.system("../trt/de"
    return


def main():
    TRAINABLE=True
    OPTIMIZE=True
    ckp = sys.argv[1]
    pb_out_path = os.path.join(os.path.dirname(ckp), "freeze_ariel.pb")
    pb_out_path_trainable = os.path.join(os.path.dirname(ckp), "freeze_ariel_trainable.pb")
    # pb_out_path_non_trainable = os.path.join(os.path.dirname(ckp), "freeze_ariel_non_trainable.pb")
    image_path = "/home/briefcam/Projects/ArielE/tf-pose-git/images/vilage.jpg"
    small_train_image = "/home/CoreNew/PoseEestimation/DataGeneration/coco_small/train/000000004554.jpg"
    ref_opt_path = "/home/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_opt_constant.pb"
    ref_model_path = "/home/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_freeze.pb"
    ref_onnx_path = os.path.join(os.path.dirname(ckp), "m_thin_1312x736_opt.onnx")
    script_model = os.path.join(os.path.dirname(ckp), "model-91000/model-91000_frozen.pb")
    script_opt_model = os.path.join(os.path.dirname(ckp), "model-91000/model-91000_frozen_opt.pb")
    onnx_model_path = os.path.join(os.path.dirname(ckp), "freeze_ariel.onnx")
    layer_name = ""#""Const"

    pre_trained_const_pb = "/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_opt_constant_432x368.pb"
    pre_trained_const_onnx = "/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_opt_constant_432x368.onnx"
    pre_trained_pb = get_ops_from_pb(pre_trained_const_pb, layer_name)
    pre_trained_onn = get_onnx_dict(pre_trained_const_onnx, layer_name)
    # compare_data_dicts(pre_trained_pb, pre_trained_onn)

    const_pb = "/briefcam / Projects / ArielE / trained_models / multi - gpu / mobilenet_thin_batch_48_lr_0.0001_432x368_gpu1_from_reg_48_129k / model - 150002 / model - 150002_frozen_opt_constant.pb".replace(" ","")
    const_onnx = "/briefcam / Projects / ArielE / trained_models / multi - gpu / mobilenet_thin_batch_48_lr_0.0001_432x368_gpu1_from_reg_48_129k / model - 150002 / model - 150002_frozen_opt_constant.onnx".replace(" ","")
    pb = get_ops_from_pb(const_pb, layer_name)
    onn = get_onnx_dict(const_onnx, layer_name)
    pairs = compare_dict_by_values(pb, onn)
    o_pairs = compare_dict_by_values(onn, pb)
    pre_pairs = compare_dict_by_values(pre_trained_pb, pre_trained_onn)
    o_pre_pairs = compare_dict_by_values(pre_trained_onn, pre_trained_pb)

    extract_heat_maps_from_pb(const_pb,"/briefcam/Projects/ArielE/tf-pose-git/images/p1.jpg")
    exit()
    #
    extract_heat_maps_from_ckp(ckp, small_train_image, trainable=False)
    # save_pb_file(ckp, pb_out_path, trainable=True, do_transforms=False)
    # extract_heat_maps_from_pb(pb_out_path, small_train_image)

    exit()

    save_pb_file(ckp, pb_out_path_trainable, trainable=True, do_transforms=False)
    saved_dict = get_ops_from_pb(pb_out_path, layer_name)
    saved_dict_trainable = get_ops_from_pb(pb_out_path_trainable, layer_name)
    compare_data_dicts(saved_dict, saved_dict_trainable)

    #
    # exit()
    # ckp_dict_trainable = get_ops_from_ckp(ckp, layer_name, trainable=True)
    # ckp_dict = get_ops_from_ckp(ckp, layer_name, trainable=False)
    # ckp_dict_2 = get_ops_from_ckp(ckp, layer_name, trainable=False)
    #
    # m,dm,s = compare_data_dicts(ckp_dict, ckp_dict)
    # m_, dm_, s_ = compare_data_dicts(ckp_dict_trainable, ckp_dict)


    # compare_data_dicts(saved_dict, ckp_dict)
    # exit()

    # create_onnx_from_pb(pb_out_path, onnx_model_path)
    # onnx_dict = get_onnx_dict(onnx_model_path, layer_name)
    # ref_onnx_dict = get_onnx_dict(ref_onnx_path, layer_name)
    ref_dict = get_ops_from_pb(ref_model_path,layer_name)
    ref_opt_dict = get_ops_from_pb(ref_opt_path, layer_name)
    script_dict = get_ops_from_pb(script_model, layer_name)
    script_opt_dict = get_ops_from_pb(script_opt_model, layer_name)


    # extract_heat_maps_from_pb(pb_out_path, image_path)


    print("Comparing")
    # compare_data_dicts(ckp_dict, ckp_dict_trainable)
    missed_layers, different_layers, same_layers = compare_data_dicts(script_dict, ckp_dict_trainable)
    print("Done")

if __name__ == '__main__':
    main()
