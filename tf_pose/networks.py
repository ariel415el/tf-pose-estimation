import os
from os.path import dirname, abspath

import tensorflow as tf
from tf_pose.network_mobilenet import MobilenetNetwork
from tf_pose.network_mobilenet_thin import MobilenetNetworkThin
from tf_pose.network_mobilenet_new import MobilenetNetworkNew

from tf_pose.network_cmu import CmuNetwork
from tf_pose.network_mobilenet_v2 import Mobilenetv2Network


def _get_base_path():
    models_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../models"
    if not os.path.exists(models_dir_path):
            models_dir_path = "./models"
    return models_dir_path
    # if not os.environ.get('OPENPOSE_MODEL', ''):
    #     return './models'
    # return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True, numHeatMaps=15, numPafMaps=26):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_fast':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_accurate':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=1.00, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable, numHeatMaps=numHeatMaps, numPafMaps=numPafMaps)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_new':
        net = MobilenetNetworkNew({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable, numHeatMaps=numHeatMaps, numPafMaps=numPafMaps,
                                     num_paf_refinements=5, num_hm_refinements=2)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'no_name_for_last_layer'

    elif type in ['mobilenet_v2_w1.4_r1.0', 'mobilenet_v2_large', 'mobilenet_v2_large_quantize']:       # m_v2_large
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, conv_width2=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.4_r0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r1.0':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w1.0_r0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w0.75_r0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.75, conv_width2=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_w0.5_r0.5' or type == 'mobilenet_v2_small':                                # m_v2_fast
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_1.4':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.4, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_1.0':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=1.0, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_0.75':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.75, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_v2_0.5':
        net = Mobilenetv2Network({'image': placeholder_input}, conv_width=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type in ['cmu', 'openpose']:
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type in ['cmu_quantize', 'openpose_quantize']:
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'train/cmu/bs8_lr0.0001_q_e80/model_latest-18000'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'

    else:
        raise Exception('Invalid Model Name.')

    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)

    return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': 'graph/cmu/graph_opt.pb',
        'openpose_quantize': 'graph/cmu/graph_opt_q.pb',
        'mobilenet_thin': 'graph/mobilenet_thin/graph_opt.pb',
        'mobilenet_v2_large': 'graph/mobilenet_v2_large/graph_opt.pb',
        'mobilenet_v2_large_r0.5': 'graph/mobilenet_v2_large/graph_r0.5_opt.pb',
        'mobilenet_v2_large_quantize': 'graph/mobilenet_v2_large/graph_opt_q.pb',
        'mobilenet_v2_small': 'graph/mobilenet_v2_small/graph_opt.pb',
    }

    base_data_dir = dirname(dirname(abspath(__file__)))
    if os.path.exists(os.path.join(base_data_dir, 'models')):
        base_data_dir = os.path.join(base_data_dir, 'models')
    else:
        base_data_dir = os.path.join(base_data_dir, 'tf_pose_data')

    graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
    if os.path.isfile(graph_path):
        return graph_path

    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
