import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_pose.pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoToolPoseDataReader
from tf_pose.pose_augment import set_network_input_wh, set_network_scale
from tf_pose.common import get_sample_images
import tf_pose.common as common
from tf_pose.networks import get_network

logger = logging.getLogger('train')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='vgg', help='model name')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--no_augmentation', action='store_false')
    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--train_dir', type=str, default='/data/public/rw/coco/annotations')
    parser.add_argument('--train_anns', type=str, default='/data/public/rw/coco/')
    parser.add_argument('--val_dir', type=str, default='/data/public/rw/coco/annotations')
    parser.add_argument('--val_anns', type=str, default='/data/public/rw/coco/')

    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--virtual_batch', type=int, default=1)
    parser.add_argument('--lr', type=str, default='0.0001')
    parser.add_argument('--decay_steps', type=int, default=10000)
    parser.add_argument('--decay_rate', type=float, default=0.33)

    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--max_iter', type=int, default=999999)

    parser.add_argument('--modelpath', type=str, default='../models/')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--quant-delay', type=int, default=-1)
    args = parser.parse_args()
    if args.gpus <= 0:
        raise Exception('gpus <= 0')
    num_heatmaps = len(common.BC_parts)
    num_pafmaps = len(common.BC_pairs) * 2
    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg'] or 'mobilenet' in args.model:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, num_pafmaps), name='vectmap')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, num_heatmaps), name='heatmap')

        # prepare data
        df = get_dataflow_batch(True, args.batchsize, images_dir=args.train_dir, anns_file=args.train_anns, augment=args.no_augmentation)
            # transfer inputs from ZMQ
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
        q_inp, q_heat, q_vect = enqueuer.dequeue()

    df_valid = get_dataflow_batch(False, args.batchsize, images_dir=args.val_dir, anns_file=args.val_anns)
    df_valid.reset_state()
    validation_cache = []

    test_images = get_sample_images(args.input_width, args.input_height)
    num_debug_images = (len(test_images) // args.batchsize) * args.batchsize
    logger.debug('### Num test image: %d/%d' % (len(test_images), num_debug_images))
    logger.debug(q_inp)
    logger.debug(q_heat)
    logger.debug(q_vect)

    # define model for multi-gpu
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect, args.gpus)

    output_vectmap = []
    output_heatmap = []
    paf_losses = []
    hm_losess = []
    last_hm_losses = []
    last_paf_losses = []
    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                print("### Network input size:  ",q_inp_split[gpu_id].shape)
                net, pretrained_backbone, last_layer = get_network(args.model, q_inp_split[gpu_id], numHeatMaps=num_heatmaps, numPafMaps=num_pafmaps)
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                outputs.append(net.get_output())

                paf_maps, heat_maps = net.loss_paf_hm()
                for idx_paf, paf_map in enumerate(paf_maps):
                    loss_paf = tf.nn.l2_loss(tf.concat(paf_map, axis=0) - q_vect_split[gpu_id], name='loss_paf_stage%d_tower%d' % (idx_paf, gpu_id))
                    paf_losses.append(loss_paf)
                    last_paf_losses.append(loss_paf)
                for idx_hm, heat_map in enumerate(heat_maps):
                    loss_hm = tf.nn.l2_loss(tf.concat(heat_map, axis=0) - q_heat_split[gpu_id], name='loss_hm_stage%d_tower%d' % (idx_hm, gpu_id))
                    hm_losess.append(loss_hm)
                    last_hm_losses.append(loss_hm)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU")):
        # define loss
        total_loss = tf.reduce_mean(paf_losses + hm_losess) / args.batchsize
        loss_last_paf = tf.reduce_mean(last_paf_losses) / args.batchsize
        loss_last_hm = tf.reduce_mean(last_hm_losses) / args.batchsize
        real_batch_size = args.batchsize * args.virtual_batch
        virtual_batch_total_loss = total_loss / real_batch_size

        # define optimizer
        num_train_images = len(os.listdir(args.train_dir))
        step_per_epoch = num_train_images // real_batch_size
        print("### Real batch size is " , real_batch_size)
        print("### Train images: %d , steps per epoch: %d"%(num_train_images, step_per_epoch))
        global_step = tf.Variable(0, trainable=False)

        # set learning rate decay
        starter_learning_rate = float(args.lr)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps=args.decay_steps, decay_rate=args.decay_rate, staircase=True)
        # learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, args.max_epoch * step_per_epoch, alpha=0.0)

    if args.quant_delay >= 0:
        logger.info('train using quantized mode, delay=%d' % args.quant_delay)
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=args.quant_delay)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if args.freeze_backbone:        
        vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Openpose') 
    print("### Optimizing %d\%d variables"%(len(vars_to_optimize),len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if args.virtual_batch < 2:
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step, var_list=vars_to_optimize, colocate_gradients_with_ops=True)
    else:
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in vars_to_optimize]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]# gradient variable list = [ (gradient,variable) ]
            gvs = optimizer.compute_gradients(virtual_batch_total_loss, var_list=vars_to_optimize, colocate_gradients_with_ops=True)
            accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
            train_step = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
            inc_gs_num = tf.assign(global_step, global_step + 1)

    print("# Defined optimizers")
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_last_paf", loss_last_paf)
    tf.summary.scalar("loss_last_hm", loss_last_hm)
    tf.summary.scalar("queue_size", enqueuer.size())

    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_last_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_last_hm = tf.placeholder(tf.float32, shape=[])
    sample_valid = tf.placeholder(tf.float32, shape=(num_debug_images, 640, 640, 3))
    valid_img = tf.summary.image('validation sample', sample_valid, num_debug_images)
    valid_loss_t = tf.summary.scalar("valid_loss", valid_loss)
    valid_loss_last_paf_t = tf.summary.scalar("valid_loss_last_paf", valid_loss_last_paf)
    valid_loss_last_hm_t = tf.summary.scalar("valid_loss_last_hm", valid_loss_last_hm)
    merged_validate_op = tf.summary.merge([ valid_img, valid_loss_t,valid_loss_last_paf_t, valid_loss_last_hm_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        training_name = '{}_batch_{}_lr_{}_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.input_width,
            args.input_height,
            args.tag
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint :
            if os.path.isdir(args.checkpoint):
                logger.info('Restore from -latest- checkpoint...')
                ckp_path = tf.train.latest_checkpoint(args.checkpoint)
            else:
                logger.info('Restore from checkpoint...')
                ckp_path = args.checkpoint
            loader = tf.train.Saver()
            loader.restore(sess, ckp_path)
            logger.info('Restore ...Done')
        elif pretrained_backbone:
            logger.info('Restore from pretrained weights... %s' % pretrained_backbone)
            if '.npy' in pretrained_backbone:
                net.load(pretrained_backbone, sess, False)
            else:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrained_backbone)
            logger.info('Restore ...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(os.path.join(args.modelpath, training_name), sess.graph)
        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()
        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        while True:
            if args.virtual_batch  < 2:
                _, gs_num = sess.run([train_op, global_step])
            else:
                sess.run(zero_ops) # zero accumulated grads
                for k in range(args.virtual_batch):
                    sess.run(accum_ops)
                sess.run([update_ops])
                sess.run([train_step, inc_gs_num])
                gs_num = global_step.eval();

            if (gs_num > step_per_epoch * args.max_epoch) or (gs_num > args.max_iter):
                break

            if gs_num - last_gs_num >= 100:
                train_loss, train_loss_last_paf, train_loss_last_hm, lr_val, summary, queue_size = sess.run([
                    total_loss, loss_last_paf, loss_last_hm, learning_rate, merged_summary_op, enqueuer.size()
                ])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                ex_per_sec = batch_per_sec * real_batch_size
                logger.info('Epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g'
                 % (gs_num / step_per_epoch, gs_num, ex_per_sec, lr_val, train_loss))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

                average_loss = average_loss_last_paf = average_loss_last_hm = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for val_images, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((val_images, heatmaps, vectmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for val_images, heatmaps, vectmaps in validation_cache:
                    lss, lss_l_paf, lss_l_hm, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, loss_last_paf, loss_last_hm, output_vectmap, output_heatmap],
                        feed_dict={q_inp: val_images, q_vect: vectmaps, q_heat: heatmaps}
                    )
                    average_loss += lss * len(val_images)
                    average_loss_last_paf += lss_l_paf * len(val_images)
                    average_loss_last_hm += lss_l_hm * len(val_images)
                    total_cnt += len(val_images)

                logger.info('Validation(%d) %s loss=%f, loss_last_paf=%f, loss_last_hm=%f'
                         % (total_cnt, training_name, average_loss / total_cnt, average_loss_last_paf / total_cnt, average_loss_last_hm / total_cnt))
                last_gs_num2 = gs_num

                # Generate test images
                outputMaps = []
                # assumes more test images than batch size
                for i in range(len(test_images) // args.batchsize) :
                    idx = i*args.batchsize
                    chunk = test_images[idx:idx+args.batchsize]
                    outputMat = sess.run(
                        outputs,
                        feed_dict={q_inp: np.array(chunk)}
                    )
                    outputMaps += [outputMat]
                outputMat = np.concatenate(outputMaps, axis=0)

                pafMat, heatMat = outputMat[:, :, :, num_heatmaps:], outputMat[:, :, :, :num_heatmaps]

                test_results = []

                for i in range(num_debug_images):
                    test_result = CocoToolPoseDataReader.display_image(test_images[i], heatMat[i], pafMat[i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)

                # save summary
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_last_paf: average_loss_last_paf / total_cnt,
                    valid_loss_last_hm: average_loss_last_hm / total_cnt,
                    sample_valid: test_results
                })
                file_writer.add_summary(summary, gs_num)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))

