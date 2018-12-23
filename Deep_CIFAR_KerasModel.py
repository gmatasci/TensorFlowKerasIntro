"""
Project Name: CNN on CIFAR dataset with Keras Model API
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: Deep_CIFAR_KerasModel.py
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
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras as krs

from utils import*


## See devices to check whether GPU or CPU are being used
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


## PARAMETERS -------------------------------------------------------------------------------------------

PARAMS = {}

PARAMS['base_dir'] = r'C:\Projects\TensorFlow'

PARAMS['tb_exp_name'] = r'CIFAR_KerasModel_TODEL'

PARAMS['data_dir'] = os.path.join(PARAMS['base_dir'], "Data", "CIFAR_10")
PARAMS['nr_classes'] = 10

PARAMS['normaliz'] = 'none'     # 'none', 'channel_standardiz', 'channel_minmax'

PARAMS['epochs'] = 200
PARAMS['batch_size_trn'] = [32, 64, 128]
# PARAMS['batch_size_trn'] = [32, 128]
PARAMS['dropout'] = [0, 0.4, 0.7]   # dropout 0 means we keep all the units
# PARAMS['dropout'] = [0.4]
PARAMS['learn_rate'] = [1e-3, 1e-4, 1e-5]     # 0.0001 usually gives the best results with Adam
# PARAMS['learn_rate'] = [1e-4]
PARAMS['patience'] = 3  # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy)

PARAMS['subsetting'] = True   # subset datasets for testing purposes


## START ---------------------------------------------------------------------

print(python_info())

print('Deep_CIFAR_KerasModel.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

start_time = tic()

## Definition of the directories
PARAMS['log_dir'] = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Tensorboard_logs")
PARAMS['model_dir'] = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Models")
PARAMS['fig_dir'] = os.path.join(PARAMS['base_dir'], PARAMS['tb_exp_name'], "Figures")

directories = [PARAMS['log_dir'], PARAMS['model_dir'], PARAMS['fig_dir']]
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
    trn_idx = np.random.choice(X_trn_img.shape[0], size=1000, replace=False)
    X_trn_img = X_trn_img[trn_idx, :, :, :]
    Y_trn = Y_trn[trn_idx, :]
    val_idx = np.random.choice(X_val_img.shape[0], size=10000, replace=False)
    X_val_img = X_val_img[val_idx, :, :, :]
    Y_val = Y_val[val_idx, :]
    tst_idx = np.random.choice(X_tst_img.shape[0], size=10000, replace=False)
    X_tst_img = X_tst_img[tst_idx, :, :, :]
    Y_tst = Y_tst[tst_idx, :]


nr_samples_trn = X_trn_img.shape[0]
nr_samples_val = X_val_img.shape[0]
nr_samples_tst = X_tst_img.shape[0]


## DEFINE CNN ------------------------------------------------------------------

def cnn_2convs(X_trn, Y_trn, X_val, Y_val, params, hparams):

    # This returns a tensor
    inputs = krs.models.Input(shape=(32, 32, 3))

    # Layer instance is callable on a tensor, and returns a tensor
    net = krs.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu')(inputs)   # with channels_last as a default, i.e., inputs with shape (batch, height, width, channels)

    ## Batch normalization by standardization of each training batch
    net = krs.layers.BatchNormalization(axis=-1, center=True, scale=True)(net)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set

    ## Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, filters]
    # Output Tensor Shape: [batch_size, 16, 16, filters]
    net = krs.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(net)

    ## Convolutional Layer #2
    net = krs.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu')(net)

    ## Batch normalization by standardization of each training batch
    net = krs.layers.BatchNormalization(axis=-1, center=True, scale=True)(net)

    ## Pooling Layer #2
    # Input Tensor Shape: [batch_size, 16, 16, filters]
    # Output Tensor Shape: [batch_size, 8, 8, filters]
    net = krs.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(net)

    ## Fully connected (dense) Layer
    # Input Tensor Shape: [batch_size, 8, 8, filters]
    # Output Tensor Shape: [batch_size, 1024]
    net = krs.layers.Flatten()(net)
    net = krs.layers.Dense(units=1024, activation='relu')(net)

    ## Dropout layer: will only be performed if training is True
    net = krs.layers.Dropout(rate=hparams['do'])(net)

    ## Final output layer producing a predicted value (logit) per each class
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, n_classes]
    probs = krs.layers.Dense(units=params['nr_classes'], activation='softmax')(net)

    # This creates a model that includes
    # the Input layer and three Dense layers  (model.summary() shows nr of trainable parameters)
    model = krs.models.Model(inputs=inputs, outputs=probs)
    adam = krs.optimizers.Adam(lr=hparams['lr'])
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hparams_str = ''
    for k, v in hparams.items():
        hparams_str += '%s_%g_' % (k, v)
    hparams_str = hparams_str[0:-1]

    log_dir_hparams = os.path.join(params['log_dir'], hparams_str)
    if os.path.exists(log_dir_hparams):
        shutil.rmtree(log_dir_hparams)
    os.makedirs(log_dir_hparams)

    earlystop_callback = krs.callbacks.EarlyStopping(monitor='val_acc', patience=params['patience'])
    tb_callback = krs.callbacks.TensorBoard(log_dir=log_dir_hparams,    # setting histogram_freq = 1 causes weird error on 2nd round of loop
                                            write_graph=True
                                            )
    best_model_path = os.path.join(params['model_dir'], 'best_model_%s.hdf5' % hparams_str)
    checkpoint_callback = krs.callbacks.ModelCheckpoint(best_model_path, monitor='val_acc',
                                                        verbose=0, save_best_only=True)

    history = model.fit(x=X_trn, y=Y_trn,
              batch_size=hparams['bs'],
              epochs=params['epochs'],
              validation_data=(X_val, Y_val),
              callbacks=[earlystop_callback, checkpoint_callback, tb_callback],
              verbose=2)  # starts training

    ## Get best accuracy and corresponding epoch
    best_val_acc = np.max(history.history['val_acc'])
    best_epoch = np.argmax(history.history['val_acc'])+1

    return best_val_acc, best_epoch, best_model_path


## Gridsearch over the hyperparameters
grid_search_list = []   # list to be converted to pd dataframe
for bs in PARAMS['batch_size_trn']:
    for lr in PARAMS['learn_rate']:
        for do in PARAMS['dropout']:
            print('Batch size = %g, Learning rate = %g, dropout = %g' % (bs, lr, do))
            hp = {}
            hp['bs'] = bs        # at each iteration reset to specific value
            hp['lr'] = lr
            hp['do'] = do
            val_OA, epoch, model_path = cnn_2convs(X_trn=X_trn_img, Y_trn=Y_trn, X_val=X_val_img, Y_val=Y_val, params=PARAMS, hparams=hp)
            grid_search_list.append({'batch_size': bs, 'learn_rate': lr, 'dropout': do,
                                     'epoch': epoch, 'val_OA': val_OA, 'model_path': model_path})  # fill row entries with dictionary

## Get best values
grid_search_df = pd.DataFrame(grid_search_list)   # convert to pd dataframe
grid_search_df.sort_values(by='val_OA', ascending=False, inplace=True)
best_bs = grid_search_df.loc[0, 'batch_size']
best_lr = grid_search_df.loc[0, 'learn_rate']
best_do = grid_search_df.loc[0, 'dropout']
best_epoch = grid_search_df.loc[0, 'epoch']
best_model_path = grid_search_df.loc[0, 'model_path']

## Load best model from saved file, as model object after .fit() is a snapshot at the "best epoch + patience" point
CNN_best = krs.models.load_model(best_model_path)

Y_tst_pred = CNN_best.predict(x=X_tst_img, verbose=1)

## From one-hot to categorical label
Y_tst_class = np.argmax(Y_tst, axis=1)
Y_tst_pred_class = np.argmax(Y_tst_pred, axis=1)

## Convert numerical labels to text labels
CIFAR10_labels = ['airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']
le = LabelEncoder()
le.fit(CIFAR10_labels)
Y_tst_class_str = le.inverse_transform(Y_tst_class)
Y_tst_pred_class_str = le.inverse_transform(Y_tst_pred_class)

## Assess test predictions and save results
RES = {}
RES['conf_mat'] = confusion_matrix(Y_tst_class, Y_tst_pred_class)
RES['OA'] = accuracy_score(Y_tst_class, Y_tst_pred_class)
RES['Kappa'] = cohen_kappa_score(Y_tst_class, Y_tst_pred_class)
RES['class_measures'] = classification_report(Y_tst_class_str, Y_tst_pred_class_str)

print('Classification results:\n\n '
      'Confusion matrix:\n %s \n\n '
      'OA=%.3f, Kappa=%.3f \n\n '
      'Class-specific measures:\n %s'
      % (RES['conf_mat'], RES['OA'], RES['Kappa'], RES['class_measures']))


print('Total ' + toc(start_time))

bla = 1


