from __future__ import division
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import scipy
import cv2
import dlib
sys.path.append('/Users/yu-chieh/seg_models/models/slim/')
sys.path.append("/Users/yu-chieh/seg_models/tf-image-segmentation/")
from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input


slim = tf.contrib.slim
cpstandard = "/Users/yu-chieh/Downloads/fcn_8s_checkpoint/model_fcn8s_final.ckpt"


def get_all_images_for_fcn(num_images):
    # get num_images images form the path and put as a matrix
    imgs = []
    num = 0
    path = '/Users/yu-chieh/Downloads/images_data_crop/'
    for f in os.listdir(path):
        if num >= num_images:
            return np.array(imgs)
        image_path = os.path.join(path,f)
        image = scipy.ndimage.imread(image_path, mode='RGB')
        # cheating version
        # image = np.dstack((image, get_xy_mask(image)))
        imgs.append(image)
        num += 1
    return np.array(imgs)

def get_facial_points(image, num_points):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    points = []
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(image, d)
        for i in range(num_points):
            pt = shape.part(i)
            points.append([int(pt.x), int(pt.y)])
    return np.array(points)

def get_xy_mask(image):
    # bad version
    image_src = image
    mask_dst = scipy.ndimage.imread('/Users/yu-chieh/Downloads/images_data_crop/02457.jpg', mode='RGB')
    dst = get_facial_points(mask_dst, 30)
    src = get_facial_points(image_src, 30)
    h, status = cv2.findHomography(src, dst)
    im_dst = cv2.warpPerspective(image_src, h, (image_src.shape[1], image_src.shape[0]))
    return im_dst

def test_fcn_featurizer(test_size, x, train_fcn=False, checkpoint_path=cpstandard):
    """
    ========== Args ==========
      checkpoint_path: Str. Path to `.npy` file containing AlexNet parameters.
       can be found here: `https://github.com/warmspringwinds/tf-image-segmentation/`
      num_channels: Int. number of channels in the input image to be featurized.
       FCN is pretrained with 3 channels.
      train_fcn: Boolean. Whether or not to train the preloaded weights.
      
    ========== Returns ==========
        A featurizer function that takes in a tensor with shape (b, h, w, c) and
        returns a tensor with shape (b, dim).
    """
    size_muliple=32
    num_class=21
    num_channels=3
    image_shape = (None, None, num_channels)  # RGB + Segmentation id
    images = tf.placeholder(tf.uint8, shape=(test_size,) + image_shape)
    preprocessed_images = tf.image.resize_images(images, size=(229, 229))

    # # Be careful: after adaptation, network returns final labels
    # # and not logits
    # with tf.variable_scope("conv_to_channel3"):
    #     filter_m = tf.Variable(tf.random_normal([1,1,num_channels,3]))
    #     preprocessed_images_3_channels = tf.nn.conv2d(preprocessed_images, filter_m, strides=[1, 1, 1, 1], padding='VALID')
    #     shape_of_this = tf.shape(preprocessed_images_3_channels)

    model = adapt_network_for_any_size_input(FCN_8s, size_muliple)
    pred, fcn_16s_variables_mapping = model(image_batch_tensor=preprocessed_images,
                                          number_of_classes=num_class,
                                          is_training=train_fcn)
    binary_pred = tf.nn.sigmoid(tf.cast(pred, tf.float32), name="sigmoid")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        # a = sess.run([shape_of_this], feed_dict={images: x})
        # print(a)
        original_imgs, output_masks = sess.run([images, binary_pred], feed_dict={images: x})
        io.imshow(original_imgs[0])
        io.show()
        io.imshow(output_masks[0].squeeze())
        io.show()

imgs = get_all_images_for_fcn(2)
print(imgs.shape)
test_fcn_featurizer(2, imgs)