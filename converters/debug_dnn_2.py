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

# def pose_resize_shortestedge(image, target_size):
#     # adjust image
#     scale = target_size / min(image.height, image.width)
#     if meta.height < meta.width:
#         newh, neww = target_size, int(scale * meta.width + 0.5)
#     else:
#         newh, neww = int(scale * meta.height + 0.5), target_size
#
#     dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
#
#     pw = ph = 0
#     if neww < _network_w or newh < _network_h:
#         pw = max(0, (_network_w - neww) // 2)
#         ph = max(0, (_network_h - newh) // 2)
#         mw = (_network_w - neww) % 2
#         mh = (_network_h - newh) % 2
#         color = random.randint(0, 255)
#         dst = cv2.copyMakeBorder(dst, ph, ph+mh, pw, pw+mw, cv2.BORDER_CONSTANT, value=(color, 0, 0))
#
#     # adjust meta data
#     adjust_skeletons = []
#     for joint in meta.skeletons:
#         adjust_joint = []
#         for point in joint:
#             if point[0] < -100 or point[1] < -100:
#                 adjust_joint.append((-1000, -1000))
#                 continue
#             # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
#             #     adjust_joint.append((-1, -1))
#             #     continue
#             adjust_joint.append((int(point[0]*scale+0.5) + pw, int(point[1]*scale+0.5) + ph))
#         adjust_skeletons.append(adjust_joint)
#
#     meta.skeletons = adjust_skeletons
#     meta.width, meta.height = neww + pw * 2, newh + ph * 2
#     meta.img = dst
#     return meta
#
#
# def get_image(path):
#     img_str = open(path, 'rb').read()
#     nparr = np.fromstring(img_str, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     ratio_w = 432 / image.width
#     ratio_h = 368 / image.height
#     ratio = max(ratio_w, ratio_h)
#     return pose_resize_shortestedge(image, int(min(image.width * ratio + 0.5, image.height * ratio + 0.5)))

def extract_heat_maps_from_ckp(ckp,image_path):
    test_images = get_sample_images(432, 368)[:1]
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(len(test_images), 368, 432, 3), name='image')

    outputs = []
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        with tf.variable_scope(tf.get_variable_scope()):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, trainable=TRAINABLE)
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
                    cv2.imwrite(os.path.join(os.path.dirname(ckp), "pb_test_%d.png" % i), test_result)
                return

def save_pb_file(ckp, path,do_transforms=False):
    with tf.Graph().as_default() as graph:
        input_node = tf.placeholder(tf.float32, shape=(1, 368, 432, 3), name='image')
        with tf.device(tf.DeviceSpec(device_type="GPU")):
            net, pretrain_path, last_layer = get_network("mobilenet_thin", input_node, None, trainable=TRAINABLE)
        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            loader = tf.train.Saver(variables_to_restore)
            loader.restore(sess, ckp)
            graph_def = tf.get_default_graph().as_graph_def()
            if do_transforms:
                transforms = ['add_default_attributes',
                              'remove_nodes(op=Identity, op=CheckNumerics)']
                              # 'strip_unused_nodes', 'sort_by_execution_order']
                              # 'fold_batch_norms', 'fold_old_batch_norms',
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


if __name__ == '__main__':
    TRAINABLE=True
    OPTIMIZE=False
    layer_name = "Openpose/MConv_Stage1_L1_1"
    ckp = sys.argv[1]
    script_model_path = "/home/CoreNew/PoseEestimation/DataGeneration/coco_full/training/bc_format/mobilenet_thin_batch_16_lr_0.01_432x368_gpu5_bc_format_/model-91000/model-91000/model-91000_frozen.pb"
    saved_model_path = os.path.join(os.path.dirname(ckp), "debug_freeze_ariel.pb")
    # ref_model_path = "/home/briefcam/Projects/ArielE/tf-pose-git/models/graph/mobilenet_thin/graph_freeze.pb"
    image_path = "/home/briefcam/Projects/ArielE/tf-pose-git/images/vilage.jpg"
    opt_model= "/home/CoreNew/PoseEestimation/DataGeneration/coco_full/training/bc_format/mobilenet_thin_batch_16_lr_0.01_432x368_gpu5_bc_format_/model-91000/debug_freeze_ariel_opt.pb"

    save_pb_file(ckp, saved_model_path, do_transforms=OPTIMIZE)
    # extract_heat_maps_from_ckp(ckp, image_path)
    extract_heat_maps_from_pb(saved_model_path, image_path)
    # extract_heat_maps_from_pb(opt_model, image_path)

    exit()

    ckp_dict = get_ops_from_ckp(ckp,layer_name)
    script_dict = get_ops_from_pb(script_model_path,layer_name)
    saved_dict = get_ops_from_pb(saved_model_path,layer_name)
    # ref_dict = get_ops_from_pb(ref_model_path,layer_name)

    missing_layers = 0
    for script_name in script_dict:
        script_val = script_dict[script_name][0]
        script_type = script_dict[script_name][1]
        if script_name not in ckp_dict:
            missing_layers += 1
            print("Op %s of type %s is missing from freezed model"%(script_name, script_type))

        # ref_val = ref_dict[script_name][0]
        # ref_type = ref_dict[script_name][1]

        ckp_val = ckp_dict[script_name][0]
        ckp_type = ckp_dict[script_name][1]

        saved_val = saved_dict[script_name][0]
        saved_type = saved_dict[script_name][1]
        if (np.sum(script_val == saved_val) != saved_val.size) or (np.sum(script_val == ckp_val) != saved_val.size):
            print("Op:script_name")
            print("\tckp type", ckp_type, end=", ")
            print("\tsaved type", saved_type,end=", ")
            print("\tscript type", script_type)
            print("\tShapes: ", script_val.shape)
            print("\tmean diff: script saved %d/%d"%(np.sum(script_val == saved_val), saved_val.size))
            print("\tmean diff: : script ckp %d/%d"%(np.sum(script_val == ckp_val), ckp_val.size))
            print("\tmean diff: : saved ckp %d/%d" % (np.sum(saved_val == ckp_val), ckp_val.size))
        # print("\tmean diff: : ref ckp %d/%d"%(np.sum(ref_val == ckp_val), ref_val.size))

        # if (np.sum(script_val == ckp_val) /  float(script_val.size) )< 1.0:
        #     print("Op: %s"%script_name)
        #     print("\tmean diff: : script ckp %d/%d"%(np.sum(script_val == ckp_val), script_val.size))
        # print("mean diff: : script saved", np.mean(script_val - saved_val))
        # print("mean diff: : ref saved", np.mean(ref_val - saved_val))
        # print("mean diff: : saved ckp", np.mean(saved_val - ckp_val))
    print("missing_layers ", missing_layers)


