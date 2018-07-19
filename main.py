from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2
import architecture
import numpy as np
import argparse


def content_loss(content, generated):
    shape = tf.shape(content)
    con_loss = tf.reduce_sum(tf.square(content - generated))
    con_loss = con_loss / (4.0 * tf.cast(shape[0] * shape[1] * shape[2], tf.float32))
    return con_loss


def gram(tensor):
    shape = tf.shape(tensor)
    tensor_reshaped = tf.reshape(tensor, [shape[0]*shape[1], shape[2]])
    return tf.matmul(tf.transpose(tensor_reshaped), tensor_reshaped)


def style_loss(style, generated):
    shape = tf.shape(style)
    gram_style = gram(style)
    gram_generated = gram(generated)
    sty_loss = tf.reduce_sum(tf.square(gram_style - gram_generated))
    sty_loss = sty_loss / ((2.0 * tf.cast(shape[0] * shape[1] * shape[2], tf.float32)) ** 2)
    return sty_loss


def main(unused_arg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--weights_path', type=str, default="vgg16.npy", help="path to the vgg16.npy file")
    parser.add_argument('-ci', '--content_image', type=str, default=None, help="path to content image")
    parser.add_argument('-si', '--style_image', type=str, default=None, help="path to style image")
    parser.add_argument('-op', '--output_path', type=str, default="art_image.png", help="path to save output")
    parser.add_argument('-lr', '--learning_rate', type=float, default=2.0, help="learning rate")
    parser.add_argument('-i', '--iterations', type=int, default=2000, help="content loss weight")
    parser.add_argument('-a', '--alpha', type=float, default=100, help="content loss weight")
    parser.add_argument('-b', '--beta', type=float, default=8, help="style loss weight")
    parser.add_argument('-lw', '--layer_loss_weights', nargs='+', type=float, default=[0.5, 1, 0.5, 0.5, 0.5], help="layer wise style loss weights")
    arg = parser.parse_args()

    if not os.path.exists(arg.weights_path):
        raise ValueError("vgg16.npy not found at {}" .format(arg.weights_path))

    if arg.content_image is None:
        raise ValueError("Path to content image not specified")
    else:
        if not os.path.exists(arg.content_image):
            raise FileNotFoundError("No image exists at the location: {}" .format(arg.content_image))
        else:
            con_img = cv2.imread(arg.content_image)

    if arg.style_image is None:
        raise ValueError("Path to content image not specified")
    else:
        if not os.path.exists(arg.style_image):
            raise FileNotFoundError("No image exists at the location: {}" .format(arg.style_image))
        else:
            sty_img = cv2.imread(arg.style_image)

    vgg_mean = [103.939, 116.779, 123.68]

    # resize content and style image to same size
    con_img = np.asarray(con_img, dtype=np.float32)
    shape = con_img.shape
    sty_img = cv2.resize(sty_img, (shape[1], shape[0]))
    sty_img = np.asarray(sty_img, dtype=np.float32)
    assert con_img.shape == sty_img.shape, "content and style images have different shape"

    # subtract mean values from each channel and reshape (required by vgg network)
    for i in range(3):
        con_img[:, :, i] = con_img[:, :, i] - vgg_mean[i]
        sty_img[:, :, i] = sty_img[:, :, i] - vgg_mean[i]
    con_img = con_img.reshape(1, shape[0], shape[1], shape[2])
    sty_img = sty_img.reshape(1, shape[0], shape[1], shape[2])

    content_image = tf.placeholder(dtype=tf.float32, shape=(1, shape[0], shape[1], shape[2]), name="content_image")
    style_image = tf.placeholder(dtype=tf.float32, shape=(1, shape[0], shape[1], shape[2]), name="style_image")
    initialize = tf.constant_initializer(value=con_img, dtype=tf.float32)
    '''random_image = tf.get_variable(initializer=tf.zeros(shape=(1, shape[0], shape[1], shape[2])), dtype=tf.float32,
                                   trainable=True, name="rnd_img")'''
    random_image = tf.get_variable(initializer=initialize, dtype=tf.float32, shape=(1, shape[0], shape[1], shape[2]),
                                   trainable=True, name="rnd_img")
    input_image = tf.concat([random_image, content_image, style_image], axis=0)

    model = architecture.Model(arg.weights_path, False)
    model.build(input_image)
    layer_1 = model.conv1_2
    layer_2 = model.conv2_2
    layer_3 = model.conv3_3
    layer_4 = model.conv4_3
    layer_5 = model.conv5_3

    con_loss = content_loss(layer_3[1, :, :, :], layer_3[0, :, :, :])
    sty_loss1 = style_loss(layer_1[2, :, :, :], layer_1[0, :, :, :])
    sty_loss2 = style_loss(layer_2[2, :, :, :], layer_2[0, :, :, :])
    sty_loss3 = style_loss(layer_3[2, :, :, :], layer_3[0, :, :, :])
    sty_loss4 = style_loss(layer_4[2, :, :, :], layer_4[0, :, :, :])
    sty_loss5 = style_loss(layer_5[2, :, :, :], layer_5[0, :, :, :])

    w = arg.layer_loss_weights
    sty_loss = sty_loss1 * w[0] + sty_loss2 * w[1] + sty_loss3 * w[2] + sty_loss4 * w[3] + sty_loss5 * w[4]
    loss = (arg.alpha * con_loss) + (arg.beta * sty_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=arg.learning_rate)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver({"rnd_img": random_image})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(arg.iterations):
            _, loss_out = sess.run([train_op, loss], feed_dict={content_image: con_img, style_image: sty_img})
            print("iteration:{}/{} loss:{}".format(i+1, arg.iterations, loss_out))

        artistic_image, = sess.run([random_image])
        artistic_image = np.squeeze(artistic_image)
        for i in range(3):
            artistic_image[:, :, i] = artistic_image[:, :, i] + vgg_mean[i]
        cv2.imwrite(arg.output_path , artistic_image)


if __name__ == "__main__":
    tf.app.run()
