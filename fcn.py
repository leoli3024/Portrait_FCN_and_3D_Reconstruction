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

def record_train_val_data(list_1, list_2):
    df = pd.DataFrame(data={"train": list_1, "val": list_2})
    df.to_csv("fcn_result.csv", sep=',',index=False)

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def main(argv=None):
    train_errors = []
    val_errors = []
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    #tf.image_summary("input_image", image, max_images=2)
    #tf.image_summary("ground_truth", tf.cast(annotation, tf.uint8), max_images=2)
    #tf.image_summary("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_images=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    #tf.scalar_summary("entropy", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    #print("Setting up summary op...")
    #summary_op = tf.merge_all_summaries()

    train_dataset_reader = BatchDatset('data/trainlist.mat')

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    #summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #if FLAGS.mode == "train":
    itr = 0
    train_images, train_annotations = train_dataset_reader.next_batch()
    trloss = 0.0
    while len(train_annotations) > 0:
        print(itr)
        #train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
        #print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
        _, rloss =  sess.run([train_op, loss], feed_dict=feed_dict)
        trloss += rloss

        if itr % 50 == 0:
            #train_loss, rpred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%f" % (itr, trloss / 100))
            train_errors.append(trloss / 100)
            trloss = 0.0
            #summary_writer.add_summary(summary_str, itr)

        if itr % 50 == 0 and itr > 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
            val_errors.append(valid_loss/100)
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
        itr += 1

        train_images, train_annotations = train_dataset_reader.next_batch()
        if itr % 100 == 0:
            saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt", itr)
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
    pred_annotation, logits = inference(image, keep_probability)
    sft = tf.nn.sigmoid(logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        feed_dict = {image: img, keep_probability: 0.5}
        rsft, pred_ann = sess.run([sft, pred_annotation], feed_dict=feed_dict)
        _, h, w, _ = rsft.shape
        preds = np.zeros((h, w, 1), dtype=np.float)
        for i in range(h):
            for j in range(w):
                if rsft[0][i][j][0] > 0.5:
                    preds[i][j][0] = 255.0
                elif rsft[0][i][j][0] > 0.2:
                    preds[i][j][0] = 128.0
                else:
                    preds[i][j]  = 0.0
        save_alpha_mask_img(preds, 'res/trimap' + '/face_demo')

def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/testlist.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        itr = 0
        test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
        #print('getting', test_annotations[0, 200:210, 200:210])
        print(len(test_images), len(test_annotations), len(test_orgs))
        while len(test_annotations) > 0:
            # if itr < 22:
            #     test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
            #     itr += 1
            #     continue
            if itr > 22:
                break
            feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 0.5}
            rsft, pred_ann = sess.run([sft, pred_annotation], feed_dict=feed_dict)
            # print(rsft.shape)
            _, h, w, _ = rsft.shape
            preds = np.zeros((h, w, 1), dtype=np.float)
            for i in range(h):
                for j in range(w):
                    if rsft[0][i][j][0] < 0.1:
                        preds[i][j][0] = 255
                    elif rsft[0][i][j][0] < 0.8:
                        preds[i][j][0] = 128
                    else:
                        preds[i][j][0]  = 0
            print(itr)
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], test_annotations[0], 'res/ann' + str(itr))
            save_alpha_mask_img(preds, 'res/trimap' + str(itr))
            save_alpha_img(test_orgs[0], pred_ann[0], 'res/pre' + str(itr))
            test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
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
    print(name)
    misc.imsave(name + '.png', amat)


def save_alpha_mask_img(mat, name):
    w, h = mat.shape[0], mat.shape[1]
    #print(mat[200:210, 200:210])
    print(w, h)
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 3), dtype=np.int)
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