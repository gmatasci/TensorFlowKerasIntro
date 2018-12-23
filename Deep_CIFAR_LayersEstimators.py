"""
Project Name: CNN on CIFAR dataset ("A Guide to TF Layers: Building a Convolutional Neural Network" tutorial)
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: Deep_CIFAR_LayerSession.py
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

sys.path.insert(0, "C:/Projects/Python/utils/")
from utils import*


## See devices:
# from tensorflow.python.client import device_lib
# local_device_protos = device_lib.list_local_devices()


## PARAMETERS -------------------------------------------------------------------------------------------

PARAMS = {}

PARAMS['base_dir'] = r'C:\Projects\Trials\TensorFlow'
PARAMS['tb_exp_name'] = r'CIFAR_LayersEstimator'

PARAMS['data_dir'] = os.path.join(PARAMS['base_dir'], "Data", "CIFAR_10")
PARAMS['nr_classes'] = 10

PARAMS['normaliz'] = 'none'     # 'none', 'channel_standardiz', 'channel_minmax'

PARAMS['epochs'] = 4
PARAMS['batch_size_trn'] = 128
# PARAMS['dropout'] = [0, 0.3, 0.5, 0.7]   # dropout 0 means we keep all the units
PARAMS['dropout'] = 0.4
# PARAMS['learn_rate'] = [1e-3, 1e-4, 1e-5]     # 0.0001 usually gives the best results with Adam
PARAMS['learn_rate'] = 1e-4
PARAMS['patience'] = 5     # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy)

PARAMS['subsetting'] = True   # subset datasets for testing purposes


## START ---------------------------------------------------------------------

print(python_info())

print('Deep_CIFAR_LayersSession.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

start_time = tic()

## Definition of the directories
log_dir = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Tensorboard_logs")
model_dir = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Models")
fig_dir = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Figures")

directories = [log_dir, model_dir, fig_dir]
for dir in directories:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

## IMPORT CIFAR DATA ---------------------------------------------------------------------------------------------------

data_batch = {}
batches = np.arange(5)+1
X_trn_raw = np.empty((0, 3072))
Y_trn = np.empty((0, 1))
for b in batches:
    data_batch["Train%s" % b] = unpickle(os.path.join(PARAMS['data_dir'], "data_batch_%d" % b))
    if b != 5:  # first 4 sets build the training set, last set is the validation set
        data_dict = data_batch["Train%s" % b]
        X_trn_raw= np.vstack( (X_trn_raw, data_dict[b"data"]) )
        Y_trn = np.vstack( ( Y_trn, np.asarray(data_dict[b"labels"]).reshape(10000,1) ) )

data_batch["Test"] = unpickle(os.path.join(PARAMS['data_dir'], "test_batch"))

## Reshape input data as a N x H x W x D tensor and output data as one hot vectors in a N x C matrix
X_trn_img = np.float32(np.transpose(np.reshape(X_trn_raw, (40000,3,32,32)), (0, 2, 3, 1)))
targets = Y_trn.reshape(-1).astype(np.int)
Y_trn = np.eye(PARAMS["nr_classes"])[targets]

X_val_raw = data_batch["Train5"][b"data"]
X_val_img = np.float32(np.transpose(np.reshape(X_val_raw, (10000,3,32,32)), (0, 2, 3, 1)))
Y_val = np.array(data_batch["Train5"][b"labels"]).reshape(10000,1)
targets = Y_val.reshape(-1).astype(np.int)
Y_val = np.eye(PARAMS["nr_classes"])[targets]

X_tst_raw = data_batch["Test"][b"data"]
X_tst_img = np.float32(np.transpose(np.reshape(X_tst_raw, (10000,3,32,32)), (0, 2, 3, 1)))
Y_tst = np.array(data_batch["Test"][b"labels"]).reshape(10000,1)
targets = Y_tst.reshape(-1).astype(np.int)
Y_tst = np.eye(PARAMS["nr_classes"])[targets]

## Normalize the 3 datasets (no norm with 'none')
if  PARAMS['normaliz'] == 'channel_standardiz':   # by channel mean computed over the entire training set
    x_mean = X_trn_img.mean(axis=(0, 1, 2), keepdims=True)
    x_std = X_trn_img.std(axis=(0, 1, 2), keepdims=True)
    X_trn_img = (X_trn_img - x_mean)/x_std
    X_val_img = (X_val_img - x_mean)/x_std
    X_tst_img = (X_tst_img - x_mean)/x_std
elif PARAMS['normaliz'] == 'channel_minmax':   # by max value in each channel (255 in 8-bit coding)
    X_trn_img = X_trn_img/255
    X_val_img = X_val_img/255
    X_tst_img = X_tst_img/255

if PARAMS['subsetting']:
    X_trn_img = X_trn_img[0:1200, :, :, :]
    Y_trn = Y_trn[0:1200, :]
    X_val_img = X_val_img[0:500, :, :, :]
    Y_val = Y_val[0:500, :]
    X_tst_img = X_tst_img[0:500, :, :, :]
    Y_tst = Y_tst[0:500, :]


nr_samples_trn = X_trn_img.shape[0]
nr_samples_val = X_val_img.shape[0]
nr_samples_tst = X_tst_img.shape[0]

steps_per_epoch = np.ceil(nr_samples_trn/PARAMS['batch_size_trn'])

## DEFINE CNN ------------------------------------------------------------------

## Define CNN with 2 conv layers
def model_fn(features, labels, mode, params):    # the dictionary params will be passed as argument to tf.estimator.Estimator()

    input_layer = features["x"]

    ## Convolutional Layer #1
    # Computes 64 features using a 5x5 filter with ReLU activation.
    # Input Tensor Shape: [batch_size, 32, 32, 3]
    # Output Tensor Shape: [batch_size, 32, 32, filters]
    with tf.name_scope('Conv_1'):
        ## With padding="same" we add as many 0 values at the border as needed to preserve the size of the input
        net = tf.layers.conv2d(input_layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    ## Batch normalization by standardization of each training batch
    with tf.name_scope('Batch_Norm_1'):
        net = tf.layers.batch_normalization(net, center=True, scale=True)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set

    ## Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, filters]
    # Output Tensor Shape: [batch_size, 16, 16, filters]
    with tf.name_scope('MaxPool_1'):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    ## Convolutional Layer #2
    with tf.name_scope('Conv_2'):
        net = tf.layers.conv2d(net, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    with tf.name_scope('Batch_Norm_2'):
        net = tf.layers.batch_normalization(net, center=True, scale=True)

    ## Pooling Layer #2
    # Input Tensor Shape: [batch_size, 16, 16, filters]
    # Output Tensor Shape: [batch_size, 8, 8, filters]
    with tf.name_scope('MaxPool_2'):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    ## Fully connected (dense) Layer
    # Input Tensor Shape: [batch_size, 8, 8, filters]
    # Output Tensor Shape: [batch_size, 1024]
    with tf.name_scope('FC_1'):
        net = tf.reshape(net, [-1, 8*8*64])
        net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)

    ## Dropout layer: will only be performed if training is True
    with tf.name_scope('Dropout'):
        net = tf.layers.dropout(net, rate=params['dropout'], training=mode==tf.estimator.ModeKeys.TRAIN)

    ## Final output layer producing a predicted value (logit) per each class
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, n_classes]
    with tf.name_scope('Softmax_output'):
        logits = tf.layers.dense(net, units=params['nr_classes'])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())   # global_step refers to the number of batches seen by the graph (keeps track of the number of batches seen so far)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

## Create the Estimator
CIFAR_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=PARAMS)

## Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}   # log the values in the "Softmax" tensor with label "probabilities"
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=steps_per_epoch)
tf.logging.set_verbosity(tf.logging.INFO)

## Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_trn_img},
    y=Y_trn,
    batch_size=PARAMS['batch_size_trn'],
    num_epochs=PARAMS['epochs'],
    shuffle=True)

## Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_val_img},
    y=Y_val,
    num_epochs=1,
    shuffle=False)

# for ep in range(PARAMS['epochs']):
start_epoch = tic()
CIFAR_classifier.train(
    input_fn=train_input_fn,
    hooks=[logging_hook])
eval_results = CIFAR_classifier.evaluate(input_fn=eval_input_fn)
print("Epoch %g (global step %g): val OA = %.2f, %s" % (PARAMS['epochs'], eval_results['global_step'], eval_results['accuracy']*100, toc(start_epoch)))


# GS_experiment = tf.contrib.learn.Experiment(
#     estimator=CIFAR_classifier,
#     train_input_fn=train_input_fn,
#     eval_input_fn=eval_input_fn,
#     eval_steps=None,
#     min_eval_frequency=np.ceil(nr_samples_trn/PARAMS['batch_size_trn']))   # to go over the dataset once only, as specified with num_epochs=1 in eval_input_fn
#
# res = GS_experiment.train_and_evaluate()







