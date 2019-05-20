#!/opt/conda/bin/python3
import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import sys
import os
import argparse
import logging
import time
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from pose_dataset import get_dataflow_batch, DataFlowToQueue, BCToolPoseDataReader
from pose_augment import PoseAugmentor

from common import get_sample_images, INPUT_OUTPUT_RATIO
import common as common
from networks import get_network
from estimator import evaluate_results, PoseEstimator, fit_humans_to_size
import debug_tools

logger = logging.getLogger('train')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet_thin', help='model name')
    parser.add_argument('--avoid_data_augmentation', action='store_false')
    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--restart_gs', action='store_true', help='restart global step when loading checkpoint')

    parser.add_argument('--val_anns', type=str)
    parser.add_argument('--train_anns', type=str)
    parser.add_argument('--limit_val_images', type=int, default=None)
    parser.add_argument('--visual_val_samples', type=int, default=30)

    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--virtual_batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--backbone_lr_factor', type=float, default=0.5)
    parser.add_argument('--decay_steps', type=int, default=10000)
    parser.add_argument('--decay_rate', type=float, default=0.33)

    parser.add_argument('--max-epoch', type=int, default=9999)
    parser.add_argument('--max_iter', type=int, default=999999)

    parser.add_argument('--modelpath', type=str, default='../models/')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--quant-delay', type=int, default=-1)
    # parse arguments
    args = parser.parse_args()
    if args.gpus <= 0:
        raise Exception('gpus <= 0')
    training_name = get_training_name(args)
    num_heatmaps = len(common.BC_parts)
    num_pafmaps = len(common.BC_pairs) * 2
    output_dir, model_dir, debug_dir =  get_debug_dirs(args.modelpath, training_name, logger)
    train_file, val_file, accuracy_file = get_debug_files(debug_dir, logger)
    virtual_batch_size = args.batchsize * args.virtual_batch

    # define dnn objective and optimzers
    enqueuer, q_inp, q_heat, q_vect, df, df_valid = get_enqueuer_and_dataflows(args.train_anns,
                                                                                args.val_anns,
                                                                                args.input_height,
                                                                                args.input_width,
                                                                                args.batchsize,
                                                                                INPUT_OUTPUT_RATIO,
                                                                                args.avoid_data_augmentation, 
                                                                                args.limit_val_images,
                                                                                num_heatmaps,
                                                                                num_pafmaps,
                                                                                logger)

    batch_average_loss, net_output_maps, pretrained_backbone_path, global_step = define_train_objective(q_inp,
                                                                                    q_heat,
                                                                                    q_vect,
                                                                                    args.batchsize,
                                                                                    args.model, 
                                                                                    num_heatmaps,
                                                                                    num_pafmaps, 
                                                                                    args.gpus, 
                                                                                    logger)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if args.virtual_batch < 2:
        train_op, exp_lr, backbone_exp_lr = get_train_op(batch_average_loss,
                                                            update_ops,
                                                            global_step,
                                                            args.lr,
                                                            args.lr*args.backbone_lr_factor,
                                                            args.decay_steps,
                                                            args.decay_rate,
         
                                                            logger)
    else:
        # we train by virtual batch size but validation is computed over real batch size
        zero_ops, accum_ops, update_ops, train_step, inc_gs_num, exp_lr, backbone_exp_lr = get_virtual_train_op(batch_average_loss / args.virtual_batch,
                                                                                                                    update_ops,
                                                                                                                    global_step,
                                                                                                                    args.lr,
                                                                                                                    args.decay_steps,
                                                                                                                    args.decay_rate,
                                                                                                                    logger)
    

    # load validation and test data
    # validation_cache = get_validation_cache(df_valid , logger)
    
    vis_val_anns_dict = get_sample_images(args.val_anns, args.input_width, args.input_height, args.batchsize, args.visual_val_samples)

    logger.info("# Defined optimizers")
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", batch_average_loss)
    tf.summary.scalar("queue_size", enqueuer.size())
    merged_train_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    num_train_images = df.size() * args.batchsize # df is a batched dataflow
    num_val_images = df_valid.size() * args.batchsize
    step_per_epoch = num_train_images // virtual_batch_size
    logger.debug("### Virtual batch size is %d"%virtual_batch_size)
    logger.debug("### Num train images: %d"%num_train_images)
    logger.debug("### Num val images: %d"%num_val_images)
    logger.debug('### Num visul val image: %d'%len(vis_val_anns_dict))
    logger.debug("### Steps per epoch: %d"%step_per_epoch)

    with tf.Session(config=config) as sess:

        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())
        load_chekpoint(sess, pretrained_backbone_path, args.checkpoint, gs=global_step if args.restart_gs else None, logger=logger)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('prepare train file writer')
        file_writer = tf.summary.FileWriter(os.path.join(output_dir), sess.graph)

        time_started = time.time()
        last_gs_num = last_gs_num2 = last_gs_num3 = 0

        initial_gs_num, train_loss = sess.run([global_step, batch_average_loss])
        logger.info('Training Started with loss: %f'%train_loss)
        # train_file.write("%d,%f,%f,%f,%d\n"%(0, train_loss, 0, 0, 0))

        while True:
            if args.virtual_batch < 2:
                _, gs_num = sess.run([train_op, global_step])
            else:
                sess.run(zero_ops) # zero accumulated grads
                for k in range(args.virtual_batch):
                    sess.run(accum_ops)
                sess.run([update_ops])
                sess.run([train_step, inc_gs_num])
                gs_num = global_step.eval()
            if (gs_num > step_per_epoch * args.max_epoch) or (gs_num > args.max_iter):
                break

            if gs_num - last_gs_num >= 100:
                train_loss, exp_lr_val, backbone_exp_lr_val, queue_size, summary = sess.run([batch_average_loss, exp_lr, backbone_exp_lr, enqueuer.size(), merged_train_summary_op])
                file_writer.add_summary(summary, gs_num)
                file_writer.flush()
                train_file.write("%d,%f,%f,%f,%d\n"%(gs_num, train_loss, exp_lr_val, backbone_exp_lr_val, queue_size))
                train_file.flush()
                debug_tools.plot_from_csv(os.path.join(debug_dir, "train_file.csv"),os.path.join(debug_dir, "train_plot.png"),  ["loss"])

                num_examples = virtual_batch_size*(gs_num - initial_gs_num)
                secs_per_ex = num_examples / (time.time() - time_started) 
                logger.info('Epoch=%.2f step=%d, loss=%g, %0.4f example sec, lr=%f, bb_lr=%f'
                            % (gs_num / step_per_epoch, gs_num, train_loss, secs_per_ex, exp_lr_val, backbone_exp_lr_val))
                last_gs_num = gs_num


            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(model_dir, 'model'), global_step=global_step)

                # Validation loss
                validation_start = time.time()
                all_batches_losses = []

                # log of test accuracy
                # for val_images, heatmaps, vectmaps in validation_cache:
                for val_images, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                    lss = sess.run(batch_average_loss,feed_dict={q_inp: val_images, q_vect: vectmaps, q_heat: heatmaps})
                    all_batches_losses += [lss]
                average_loss = np.mean(all_batches_losses)
                logger.info('Validation %s loss=%f total time. %f' % (args.tag, average_loss, time.time() - validation_start))

                last_gs_num2 = gs_num

                # save validation summary
                val_file.write("%d,%f\n"%(gs_num, average_loss))
                val_file.flush()
                debug_tools.plot_from_csv(os.path.join(debug_dir, "val_file.csv"), os.path.join(debug_dir, "val_plot.png"), ["loss"])
                debug_tools.plot_combined_train_val(os.path.join(debug_dir, "train_file.csv"),os.path.join(debug_dir, "val_file.csv"), os.path.join(debug_dir, "combined_plot.png"), ["loss"])

            if gs_num - last_gs_num3 >= 5000 and args.visual_val_samples > 0:
                test_start = time.time()
                last_gs_num3 = gs_num
                try:
                    recall, precision, collages_images, acc_images = visual_debug_network(sess, q_inp, net_output_maps, vis_val_anns_dict,args.batchsize, args.input_width, args.input_height, num_heatmaps, logger)
                except Exception as e:
                    print("Skipping visual validation due to an error: output shape")
                    print(e)
                    continue
                accuracy_file.write("%d,%f,%f\n"%(gs_num, recall, precision))
                accuracy_file.flush()
                debug_tools.plot_from_csv(os.path.join(debug_dir, "acc_file.csv"), os.path.join(debug_dir, "acc_plot.png"), ["recall", "percision"])
                output_dir = os.path.join(debug_dir, 'visual-val_step_%d_recall_%f_precision_%f'%(gs_num, recall, precision))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for idx in range(len(collages_images)):
                    cv2.imwrite(os.path.join(output_dir, "collage_%d.png"%idx), cv2.cvtColor(collages_images[idx].astype(np.float32), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, "accuracy_%d.png"%idx),  cv2.cvtColor(acc_images[idx].astype(np.float32), cv2.COLOR_RGB2BGR))
                logger.info('Test total time. %f' % (time.time() - test_start))
        train_file.close()
        val_file.close()
        accuracy_file.close()
        file_writer.close()
        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

    logger.info('optimization finished. %f' % (time.time() - time_started))

def get_training_name(args):
    training_name = '{}_b_{}x{}_lr_{:.5f}-{:.5f}x{}x{}_r_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.virtual_batch,
            args.lr,
            args.lr*args.backbone_lr_factor,
            args.decay_rate,
            args.decay_steps,
            args.input_width,
            args.input_height,
            args.tag
        )
    return training_name

def load_chekpoint(sess, pretrained_backbone_path, checkpoint_path, gs, logger):
    if checkpoint_path :
        if os.path.isdir(checkpoint_path):
            logger.info('Restore from -latest- checkpoint...')
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            logger.info('Restore from checkpoint...')
        loader = tf.train.Saver()
        loader.restore(sess, checkpoint_path)
        logger.info('Restore ...Done')
    elif pretrained_backbone_path:
        logger.info('Restore from pretrained weights... %s' % pretrained_backbone_path)

        loader = tf.train.Saver(bb_restorable_variables())
        loader.restore(sess, pretrained_backbone_path)
        logger.info('Restore ...Done')

    if gs is not None:
        reset_gs  = gs.assign(1)
        sess.run(reset_gs)

def get_validation_cache(df_valid , logger):
    validation_cache = []
    logger.info("Loading validation Cach")
    for val_images, heatmaps, vectmaps in tqdm(df_valid.get_data()):
        validation_cache.append((val_images, heatmaps, vectmaps))
    logger.info("Deleting df_valid")
    logger.info("Validation cache loaded: %d real batchs"%len(validation_cache))
    df_valid.reset_state()
    del df_valid
    df_valid = None
    return validation_cache

def define_train_objective(q_inp, q_heat, q_vect, batchsize, model_type,num_heatmaps, num_pafmaps, num_gpus, logger):
    losses = [] # for mobilenet_thin
    paf_losses = []
    hm_losses = []
    net_output_maps = []
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, num_gpus), tf.split(q_heat, num_gpus), tf.split(q_vect, num_gpus)
    for gpu_id in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrained_backbone_path, last_layer = get_network(model_type, q_inp_split[gpu_id], numHeatMaps=num_heatmaps, numPafMaps=num_pafmaps)
                vect, heat = net.loss_last()
                net_output_maps.append(net.get_output())

                paf_maps, heat_maps = net.loss_paf_hm()
                if model_type == "mobilenet_thin":
                    for idx, (paf_map, heat_map) in enumerate(zip(paf_maps, heat_maps)):
                        loss_paf = tf.nn.l2_loss(tf.concat(paf_map, axis=0) - q_vect_split[gpu_id], name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                        loss_hm = tf.nn.l2_loss(tf.concat(heat_map, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                        losses.append(tf.reduce_mean([loss_paf, loss_hm]))
                if model_type == "mobilenet_new":
                    for idx_paf, paf_map in enumerate(paf_maps):
                        loss_paf = tf.nn.l2_loss(tf.concat(paf_map, axis=0) - q_vect_split[gpu_id], name='loss_paf_stage%d_tower%d' % (idx_paf, gpu_id))
                        paf_losses.append(loss_paf)
                    for idx_hm, heat_map in enumerate(heat_maps):
                        loss_hm = tf.nn.l2_loss(tf.concat(heat_map, axis=0) - q_heat_split[gpu_id], name='loss_hm_stage%d_tower%d' % (idx_hm, gpu_id))
                        hm_losses.append(loss_hm)
    # merge batch results from different GPUS
    net_output_maps = tf.concat(net_output_maps, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU")): # TODO : was GPU
        # define loss
        if model_type == "mobilenet_thin":
            batch_average_loss = tf.reduce_sum(losses) 
        if model_type == "mobilenet_new":
            batch_average_loss = tf.reduce_mean([tf.reduce_sum(paf_losses), tf.reduce_sum(hm_losses)])

    batch_average_loss = batch_average_loss / batchsize
    return batch_average_loss  , net_output_maps, pretrained_backbone_path, tf.Variable(0, trainable=False)


def get_train_op(loss_func, update_ops, gs_var, pose_lr, bb_lr,decay_steps, decay_rate,  logger):
    pose_vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Openpose')
    backbone_vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'MobilenetV1')

    exp_lr = tf.train.exponential_decay(pose_lr, gs_var, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    backbone_exp_lr = tf.train.exponential_decay(bb_lr, gs_var, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(exp_lr, epsilon=1e-8)
    optimizer_backbone = tf.train.AdamOptimizer(backbone_exp_lr, epsilon=1e-8)
    with tf.control_dependencies(update_ops):
        train_op_pose = optimizer.minimize(loss_func, gs_var, var_list=pose_vars_to_optimize, colocate_gradients_with_ops=True)
        train_op_backbone = optimizer_backbone.minimize(loss_func, gs_var, var_list=backbone_vars_to_optimize, colocate_gradients_with_ops=True)
        train_op = tf.group(train_op_pose, train_op_backbone)
        logger.info("### Optimizing %d vats with lr %f and  %d vars with lr %f"%(len(pose_vars_to_optimize), pose_lr ,len(backbone_vars_to_optimize), bb_lr))
    return train_op, exp_lr, backbone_exp_lr

def get_virtual_train_op(loss_func, update_ops, gs_var, lr, decay_steps, decay_rate, logger):
    all_vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    exp_lr = tf.train.exponential_decay(lr, gs_var, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(exp_lr, epsilon=1e-8)
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in all_vars_to_optimize]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]# gradient variable list = [ (gradient,variable) ]
    gvs = optimizer.compute_gradients(loss_func, var_list=all_vars_to_optimize, colocate_gradients_with_ops=True)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    train_step = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
    inc_gs_num = tf.assign(gs_var, gs_var + 1)
    logger.info("### Optimizing %d vars with lr %f"%(len(all_vars_to_optimize), lr))
    return zero_ops, accum_ops, update_ops, train_step, inc_gs_num, exp_lr, exp_lr

def get_debug_files(debug_dir,logger):
    train_file = open(os.path.join(debug_dir, "train_file.csv"), "w")
    val_file = open(os.path.join(debug_dir, "val_file.csv"), "w")
    accuracy_file = open(os.path.join(debug_dir, "acc_file.csv"), "w")
    train_file.write("Step_number,loss,exp_lr_val,backbone_exp_lr_val,queue_size\n")
    val_file.write("Step_number,loss\n")
    accuracy_file.write("Step_number,recall,percision\n")
    return train_file, val_file, accuracy_file


def get_debug_dirs(output_dir, training_name, logger):
    logger.info('Creating output files and folders.')
    output_dir = os.path.join(output_dir, training_name)
    debug_dir = os.path.join(output_dir, "debug")
    model_dir = os.path.join(output_dir, "ckps")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    return output_dir, model_dir, debug_dir

def bb_restorable_variables():
    vs = {v.op.name: v for v in tf.global_variables() if
          'MobilenetV1/Conv2d' in v.op.name and
          # 'global_step' not in v.op.name and
          # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
          'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
          'Ada' not in v.op.name and 'Adam' not in v.op.name
          }
    return vs

def get_enqueuer_and_dataflows(train_anns, val_anns, input_height, input_width, batchsize, input_output_ratio, avoid_data_augmentation, limit_val_images, num_heatmaps, num_pafmaps, logger):
    pose_augmentor = PoseAugmentor(input_width, input_height, input_output_ratio)
    output_w, output_h = input_width // input_output_ratio, input_height // input_output_ratio
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(batchsize, input_height, input_width, 3), name='image')
        vectmap_node = tf.placeholder(tf.float32, shape=(batchsize, output_h, output_w, num_pafmaps), name='vectmap')
        heatmap_node = tf.placeholder(tf.float32, shape=(batchsize, output_h, output_w, num_heatmaps), name='heatmap')
        # prepare data
        df = get_dataflow_batch(train_anns, is_train=True, batchsize=batchsize, augmentor=pose_augmentor, augment=avoid_data_augmentation)
            # transfer inputs from ZMQ
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
        q_inp, q_heat, q_vect = enqueuer.dequeue()

    df_valid = get_dataflow_batch(val_anns, is_train=False, batchsize=batchsize, augmentor=pose_augmentor, augment=False, limit_data=limit_val_images)
    df_valid.reset_state()

    return enqueuer, q_inp, q_heat, q_vect, df, df_valid

def visual_debug_network(sess, q_inp,  net_output_maps, vis_val_anns_dict, batchsize, input_width, input_height, num_heatmaps, logger):

    # Test accuracy, and debug images
    outputMaps = []
    # assumes more test images than batch size
    
    keys = list(vis_val_anns_dict)
    visual_val_images = [common.read_imgfile(k,input_width, input_height) for k in keys]
    visual_val_anns_list = [vis_val_anns_dict[k] for k in keys]
    for i in tqdm(range(int(len(visual_val_images) / batchsize))):
        idx = i*batchsize
        chunk_images = visual_val_images[idx:idx+batchsize]
        batch_output = sess.run(
            net_output_maps,
            feed_dict={q_inp: np.array(chunk_images)}
        )
        outputMaps += [batch_output]
    outputMat = np.concatenate(outputMaps, axis=0)

    pafMat, heatMat = outputMat[:, :, :, num_heatmaps:], outputMat[:, :, :, :num_heatmaps]

    logger.info("Evaluate_results on  %d images"%len(visual_val_images))
    return evaluate_results(visual_val_images,
                            heatMat,
                            pafMat,
                            keys,
                            visual_val_anns_list)

if __name__ == '__main__':
    main()