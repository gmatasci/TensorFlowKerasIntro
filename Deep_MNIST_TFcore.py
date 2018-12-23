"""
Project Name: CNN on MNIST dataset (Deep MNIST for Experts tutorial)
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: Deep_MNIST.py
Objective:
"""

## IMPORT MODULES ---------------------------------------------------------------

import os
import sys
import glob
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import tensorflow as tf

from utils import*
from conviz import*

if __name__ == '__main__':

## PARAMETERS -------------------------------------------------------------------------------------------

    PARAMS = {}

    PARAMS['base_dir'] = r'C:\Projects\TensorFlow'


## START ---------------------------------------------------------------------

    print(python_info())

    print('Deep_MNIST_TFcore.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()


## IMPORT MNIST DATA ---------------------------------------------------------------------------------------------------

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

## SOFTMAX LINEAR REGRESSION MODEL -------------------------------------------------------------------------------------

    ## Placeholders for input data
    x = tf.placeholder(tf.float32, shape=[None, 784])   # None indicates the number of rows, i.e., the batch size that can be of any size
    y_ = tf.placeholder(tf.float32, shape=[None, 10])   # one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.

    W = tf.Variable(tf.zeros([784, 10]))   # because we have 784 input features and 10 outputs (the classes)
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # internally applies the softmax on the model's unnormalized model prediction and sums across all classes
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # to add new operations to the computation graph (compute gradients, compute parameter update steps, and apply update steps to the parameters)

    ## The Operation train_step will apply the gradient descent updates to the parameters, so we run it in a loop
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    ## To visualize the images
    # for im in np.arange(10):
    #     plt.figure()
    #     plt.imshow(batch[0][im,].reshape(28, 28))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Test accuracy
    print('Linear regression: test accuracy = %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))   # test images are (10000 x 784), labels are (10000 x 10)

## DEFINE MULTILAYER CNN ------------------------------------------------------------------------------------------------------

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    ## Convolution with stride 1
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    ## Classic 2x2 max pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    ## 1st CONV layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 2nd and 3rd dimensions corresponding to image width and height, and 4th imension corresponding to the number of input color channels
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32])   # 5 x 5 kernel size, 1 input channel, 32 output channels
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])  # one bias for each output channel
        variable_summaries(b_conv1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)   # 14x14x32 output

    ## Add weights tensor to collection to retrieve weights afterwards
    tf.add_to_collection('conv_weights', W_conv1)
    ## Add output to collection
    tf.add_to_collection('conv_output', h_conv1)

    ## 2nd CONV layer
    W_conv2 = weight_variable([5, 5, 32, 64])   # 5 x 5 kernel size, 32 input channel, 64 output channels
    b_conv2 = bias_variable([64])  # one bias for each output channel
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)   # 7x7x64 output

    tf.add_to_collection('conv_output', h_conv2)

    ## Fully-connected layer (with 1024 neurons)
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        h_fc1 = tf.nn.relu(preactivate)
        tf.summary.histogram('pre_activations', preactivate)

    ## Add dropout to this last layer to reduce overfitting
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## Output layer with softmax regression
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ## Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(PARAMS['base_dir'], sess.graph)
    test_writer = tf.summary.FileWriter(PARAMS['base_dir'])
    tf.global_variables_initializer().run()


## TRAIN AND TEST MULTILAYER CNN ------------------------------------------------------------------------------------------------------

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    ## tf session destroyed when with block is exited
    steps = 50
    batch_size = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch = mnist.train.next_batch(batch_size)
            if i % 50 == 0:

                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_writer.add_summary(summary, i)

                # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

                val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
                print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy, val_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})   # update parameters

        test_acc_final = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('CNN: test accuracy (%g steps, batch size = %g) = %g' % (steps, batch_size, test_acc_final))

        # get weights of all convolutional layers
        # no need for feed dictionary here
        conv_weights = sess.run([tf.get_collection('conv_weights')])
        for i, c in enumerate(conv_weights[0]):
            plot_conv_weights(c, 'conv{}'.format(i), PARAMS['base_dir'])

        # get output of all convolutional layers
        # here we need to provide an input image
        conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={x: mnist.test.images[5,].reshape(1, 784)})
        for i, c in enumerate(conv_out[0]):
            plot_conv_output(c, 'conv{}'.format(i), PARAMS['base_dir'])


print('Total ' + toc(start_time))



