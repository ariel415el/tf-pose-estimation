from __future__ import absolute_import

import tensorflow as tf

from tf_pose import network_base
import tf_pose.common


class MobilenetNetworkNew(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=None, numHeatMaps=len(tf_pose.common.BC_parts), numPafMaps=2*len(tf_pose.common.BC_pairs), num_paf_refinements=5, num_hm_refinements=0):
        self.num_paf_refinements = num_paf_refinements
        self.num_hm_refinements = num_hm_refinements
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        self.numHeatMaps = numHeatMaps
        self.numPafMaps = numPafMaps
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('image')
             .convb(3, 3, depth(32), 2, name='Conv2d_0')
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1')
             .separable_conv(3, 3, depth(128), 2, name='Conv2d_2') #strided
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3')
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4') #strided
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_8')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_9')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_10')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_11')
             # .separable_conv(3, 3, depth(1024), 2, name='Conv2d_12')
             # .separable_conv(3, 3, depth(1024), 1, name='Conv2d_13')
             )

        (self.feed('Conv2d_3').max_pool(2, 2, 2, 2, name='Conv2d_3_pool'))

        (self.feed('Conv2d_3_pool', 'Conv2d_7', 'Conv2d_11')
         .concat(3, name='feat_concat'))

        feature_lv = 'feat_concat'
        with tf.variable_scope(None, 'Openpose'):
            prefix_paf = 'Paf_refinement_1'
            (self.feed(feature_lv)
             .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix_paf + '_4')
             .separable_conv(1, 1, self.numPafMaps, 1, relu=False, name=prefix_paf + '_paf'))

            for paf_stage_id in range(self.num_paf_refinements):
                last_paf = 'Paf_refinement_%d_paf' % (paf_stage_id + 1)
                prefix_paf = 'Paf_refinement_%d' % (paf_stage_id + 2)
                (self.feed(feature_lv, last_paf)
                 .concat(3, name=prefix_paf + '_concat')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_1')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_2')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix_paf + '_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix_paf + '_4')
                 .separable_conv(1, 1, self.numPafMaps, 1, relu=False, name=prefix_paf + '_paf'))

            last_paf = 'Paf_refinement_%s_paf'%(self.num_paf_refinements + 1)
            prefix_hm = 'Hm_refinement_1'
            (self.feed(feature_lv, last_paf)
             .concat(3, name=prefix_hm + '_concat')
             .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_3')
             .separable_conv(1, 1, depth2(128), 1, name=prefix_hm + '_4')
             .separable_conv(1, 1, self.numHeatMaps, 1, relu=False, name=prefix_hm + '_hm'))

            for hm_stage_id in range(self.num_hm_refinements):
                last_hm = 'Hm_refinement_%d_hm' % (hm_stage_id + 1)
                prefix_hm = 'Hm_refinement_%d' % (hm_stage_id + 2)
                (self.feed(feature_lv, last_paf , last_hm)
                .concat(3, name=prefix_hm + '_concat')
                .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_1')
                .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_2')
                .separable_conv(3, 3, depth2(128), 1, name=prefix_hm + '_3')
                .separable_conv(1, 1, depth2(128), 1, name=prefix_hm + '_4')
                .separable_conv(1, 1, self.numHeatMaps, 1, relu=False, name=prefix_hm + '_hm'))


            # final result
            last_hm = 'Hm_refinement_%d_hm' % (self.num_hm_refinements + 1)
            (self.feed(last_hm, last_paf)
             .concat(3, name='concat_stage7'))

    def loss_paf_hm(self):
        pafs = []
        hms = []
        for layer_name in sorted(self.layers.keys()):
            if '_paf' in layer_name:
                pafs.append(self.layers[layer_name])
            if '_hm' in layer_name:
                hms.append(self.layers[layer_name])

        return pafs, hms

    def loss_last(self):
        last_pf = "Paf_refinement_%s_paf"%(self.num_paf_refinements+1)
        last_hm = "Hm_refinement_%s_hm"%(self.num_hm_refinements+1)
        return self.get_output(last_pf), self.get_output(last_hm)

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        return vs
