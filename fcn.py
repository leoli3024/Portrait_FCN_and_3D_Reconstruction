from __future__ import division
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import scipy
# import cv2
# import dlib
import pandas as pd
from PIL import Image
import csv
sys.path.append('/Users/yu-chieh/seg_models/models/slim/')
from portrait_plus import BatchDatset, TestDataset
import TensorflowUtils_plus as utils
from scipy import misc
import datetime

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

slim = tf.contrib.slim
cpstandard = "/Users/yu-chieh/Downloads/fcn_16s_checkpoint/model_fcn16s_final.ckpt"


### SMART NETWORK
"""
    large portion derived from taken from https://github.com/PetroWu/AutoPortraitMatting/blob/master/FCN_plus.py
"""
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        if name in ['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4']:
            continue
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def myvgg(image):
    padded_input = tf.pad(image, [[0, 0], [100, 100], [100, 100], [0, 0]], "CONSTANT")
    conv1_1 = tf.layers.conv2d(
          inputs=padded_input,
          filters=64,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input1_2 = tf.pad(conv1_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv1_2 = tf.layers.conv2d(
          inputs=padded_input1_2,
          filters=64,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

    padded_input2_1 = tf.pad(pool1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv2_1 = tf.layers.conv2d(
          inputs=padded_input2_1,
          filters=128,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))

    padded_input2_2 = tf.pad(conv2_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv2_2 = tf.layers.conv2d(
          inputs=padded_input2_2,
          filters=128,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
    padded_input3_1 = tf.pad(pool2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv3_1 = tf.layers.conv2d(
          inputs=padded_input3_1,
          filters=256,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input3_2 = tf.pad(conv3_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv3_2 = tf.layers.conv2d(
          inputs=padded_input3_2,
          filters=256,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input3_3 = tf.pad(conv3_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv3_3 = tf.layers.conv2d(
          inputs=padded_input3_3,
          filters=256,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)
    padded_input4_1 = tf.pad(pool3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv4_1 = tf.layers.conv2d(
          inputs=padded_input4_1,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input4_2 = tf.pad(conv4_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv4_2 = tf.layers.conv2d(
          inputs=padded_input4_2,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input4_3 = tf.pad(conv4_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv4_3 = tf.layers.conv2d(
          inputs=padded_input4_3,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)

    padded_input5_1 = tf.pad(pool4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv5_1 = tf.layers.conv2d(
          inputs=padded_input5_1,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input5_2 = tf.pad(conv5_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv5_2 = tf.layers.conv2d(
          inputs=padded_input5_2,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    padded_input5_3 = tf.pad(conv5_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    conv5_3 = tf.layers.conv2d(
          inputs=padded_input5_3,
          filters=512,
          kernel_size=3,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))

    net = dict()
    net['conv5_3'] = conv5_3
    net['pool4'] = pool4
    net['pool3'] = pool3
    return net





def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]

    weights = np.squeeze(model_data['layers'])

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def myinference_pretrained_weights(image, keep_prob):
    # print("setting up vgg initialized conv layers ...")
    # model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    # mean = model_data['normalization'][0][0][0]
    # weights = np.squeeze(model_data['layers'])
    with tf.variable_scope("inference"):
        # image_net = vgg_net(weights, image)
        image_net = myvgg(image)
        conv_final_layer = image_net["conv5_3"]
        pool5 = tf.layers.max_pooling2d(conv_final_layer, 2, 2)
        conv6 = tf.layers.conv2d(
          inputs=pool5,
          filters=4096,
          kernel_size=7,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        relu_dropout6 = tf.nn.dropout(conv6, keep_prob=keep_prob)
        conv7 = tf.layers.conv2d(
          inputs=relu_dropout6,
          filters=4096,
          kernel_size=1,
          padding="valid",
          activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        if FLAGS.debug:
            utils.add_activation_summary(conv7)
        relu_dropout7 = tf.nn.dropout(conv7, keep_prob=keep_prob)

        #### first deconv
        score = tf.layers.conv2d(
          inputs=relu_dropout7,
          filters=2,
          padding="valid",
          kernel_size=1,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        # score2
        conv_t1 = tf.layers.conv2d_transpose(
                    inputs=score,
                    filters=2,
                    padding="valid",
                    kernel_size=4,
                    strides=2,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        score_pool4 = tf.layers.conv2d(
          inputs=image_net["pool4"],
          filters=2,
          kernel_size=1,
          padding="valid",
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        score_fused = utils.crop_and_add(score_pool4, conv_t1)

        #### second deconv
        # score4
        conv_t2 = tf.layers.conv2d_transpose(
                    inputs=score_fused,
                    filters=2,
                    padding="valid",
                    kernel_size=4,
                    strides=2,
                    use_bias=False,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        score_pool3 = tf.layers.conv2d(
          inputs=image_net["pool3"],
          filters=2,
          kernel_size=1,
          padding="valid",
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        score_fused2 = utils.crop_and_add(score_pool3, conv_t2)
        # ### final deconv
        # # upsample
        conv_t3 = tf.layers.conv2d_transpose(
                    inputs=score_fused2,
                    filters=2,
                    padding="valid",
                    kernel_size=16,
                    strides=8,
                    use_bias=False,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        mask = utils.crop_and_add(conv_t3, image, to_add=False)
        # this is not needed
        annotation_pred = tf.argmax(mask, dimension=3, name="prediction")
    return annotation_pred, mask


def record_train_val_data(list_1, list_2):

    df = pd.DataFrame(data={"train": list_1, "val": list_2})
    df.to_csv(str(datetime.datetime.now()) + "fcn_result.csv", sep=',',index=False)

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    # if FLAGS.debug:
    #     # print(len(var_list))
    #     for grad, var in grads:
    #         utils.add_gradient_summary(grad, var)
    return optimizer.minimize(loss_val)

def main(argv=None):
    batch_size=1
    train_errors = []
    val_errors = []
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    # pred_annotation, logits = inference(image, keep_probability)
    _, logits = myinference_pretrained_weights(image, keep_probability)
    loss_to_record = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3])))
    loss = loss_to_record + tf.reduce_mean(tf.losses.get_regularization_loss())
    #tf.scalar_summary("entropy", loss)
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)
    train_dataset_reader = BatchDatset('data/trainlist.mat', "train", batch_size)
    validation_dataset_reader = TestDataset('data/testlist.mat', batch_size)
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    itr = 0
    train_images, train_annotations = train_dataset_reader.next_batch()
    valid_images, valid_annotations, _ = validation_dataset_reader.next_batch()
    try:
        while itr < 7000:
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
            _, rloss =  sess.run([train_op, loss_to_record], feed_dict=feed_dict)
            print(rloss)
            if itr % 10 == 0 and itr > 0:
                #train_loss, rpred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%f" % (itr, rloss))
                train_errors.append(rloss)
                #summary_writer.add_summary(summary_str, itr)

            if itr % 10 == 0 and itr > 0:
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                               keep_probability: 1.0})
                val_errors.append(valid_loss)
                print("%d ---> Validation_loss: %g" % (itr, valid_loss))
            if itr % 500 == 0 and itr > 0:
                print("saving checkpoint")
                saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt", itr)
            itr += 1
            train_images, train_annotations = train_dataset_reader.next_batch()
            valid_images, valid_annotations, _ = validation_dataset_reader.next_batch()
            # reloop
            if len(valid_images) <= 0:
                print("reset validation set")
                validation_dataset_reader = TestDataset('data/testlist.mat', batch_size)
                valid_images, valid_annotations, _ = validation_dataset_reader.next_batch()
            if len(train_annotations) <= 0:
                train_dataset_reader = BatchDatset('data/trainlist.mat', "train", batch_size)
                train_images, train_annotations = train_dataset_reader.next_batch()
    except:
        print("save session and data")
        saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt")
        record_train_val_data(train_errors, val_errors)
        sys.exit()
    saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt")
    record_train_val_data(train_errors, val_errors)

    '''elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)'''

def pred_one_image(img):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")
    pred_annotation, logits = myinference_pretrained_weights(image, keep_probability)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        feed_dict = {image: img, keep_probability: 1}
        _, logits = sess.run([pred_annotation, logits], feed_dict=feed_dict)
        difference = np.exp(logits[:, :, 1] - logits[:, :, 0])
        final = difference/(1+difference)
        final = (final > 0.5).astype(float)
        save_alpha_mask_img(final, 'res/trimap' + '/face_demo')

def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = myinference_pretrained_weights(image, keep_probability)
    test_dataset_reader = TestDataset('data/testlist.mat')
    # train_dataset_reader = BatchDatset('data/trainlist.mat', "train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        itr = 0
        test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
        # train_images, train_annotations = train_dataset_reader.next_batch()
        while len(test_annotations) > 0:
            if itr > 22:
                break
            feed_dict = {image: test_images, keep_probability: 1}
            _, l = sess.run([pred_annotation, logits], feed_dict=feed_dict)
            print(l.shape, "l")
            l = np.squeeze(l)
            difference = np.exp(l[:, :, 1] - l[:, :, 0])
            # print(difference.shape)
            final = (difference / (1.0+difference))
            final[final <= 0.5] = 0
            # final[((final > 0.5) & (final <= 0.8))] = 128
            final[final > 0.5] = 255
            trimap = final
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], test_annotations[0], 'res/ann' + str(itr))
            save_alpha_img(test_orgs[0], trimap, 'res/trimap' + str(itr))

            # save_alpha_mask_img(trimap, 'res/trimap' + str(itr))
            test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
            # train_images, train_annotations = train_dataset_reader.next_batch()
            itr += 1

def save_alpha_img(org, mat, name):
    w, h = mat.shape[0], mat.shape[1]
    #print(mat[200:210, 200:210])
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = np.round(rmat * 1000)
    amat[:, :, 0:3] = org
    #print(amat[200:205, 200:205])
    #im = Image.fromarray(np.uint8(amat))
    #im.save(name + '.png')
    # print(name)
    misc.imsave(name + '.png', amat)


def save_alpha_mask_img(mat, name):
    # print(mat.shape)
    w, h = mat.shape[0], mat.shape[1]
    #print(mat[200:210, 200:210])
    # print(w, h)
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 3), dtype=np.uint8)
    amat[:, :, 0] = rmat
    amat[:, :, 1] = rmat
    amat[:, :, 2] = rmat
    # amat[:, :, 0:3] = org
    #print(amat[200:205, 200:205])
    # im = Image.fromarray(np.uint8(amat))
    # im.save(name + '.png')
    misc.imsave(name + '.png', amat)

### call main to train, pred to predict ### 
main()

# image = TestDataset('data/testlist.mat').get_images(20)[0]
# image = np.expand_dims(image, axis=0)
# print(image)
# pred_one_image(image)
# pred()