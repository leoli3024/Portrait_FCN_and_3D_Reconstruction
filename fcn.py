from __future__ import division
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import scipy
sys.path.append('/Users/yu-chieh/seg_models/models/slim/')
sys.path.append("/Users/yu-chieh/seg_models/tf-image-segmentation/")
from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input


slim = tf.contrib.slim
cpstandard = "/Users/yu-chieh/Downloads/fcn_8s_checkpoint/model_fcn8s_final.ckpt"


def make_fcn_featurizer(batch_size, x, train_fcn=False, checkpoint_path=cpstandard):
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
    images = tf.placeholder(tf.uint8, shape=(batch_size,) + image_shape)
    preprocessed_images = tf.image.resize_images(images, size=(229, 229))

    # Be careful: after adaptation, network returns final labels
    # and not logits
    model = adapt_network_for_any_size_input(FCN_8s, size_muliple)
    
    pred, fcn_16s_variables_mapping = model(image_batch_tensor=preprocessed_images,
                                          number_of_classes=num_class,
                                          is_training=train_fcn)

    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        original_imgs, output_masks = sess.run([images, pred], feed_dict={images: x})
        io.imshow(original_imgs[0])
        io.show()
        
        io.imshow(output_masks[0].squeeze())
        io.show()


image = scipy.ndimage.imread('/Users/yu-chieh/Desktop/Sierra.png', mode='RGB')
image = np.resize(image, (1,) + image.shape)
print(image.shape)

make_fcn_featurizer(1, image)
# make_fcn_featurizer(1, np.zeros((1, 229, 229, 3)))

