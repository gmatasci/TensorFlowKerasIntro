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
PARAMS['tb_exp_name'] = r'CIFAR_LayersSession'
PARAMS['data_dir'] = os.path.join(PARAMS['base_dir'], "Data", "CIFAR_10")
PARAMS['nr_classes'] = 10
PARAMS['dev_placement'] = False    # whether to plot device placement (CPU or GPU) of the various tensors

PARAMS['normaliz'] = 'none'     # 'none', 'channel_standardiz', 'channel_minmax'

PARAMS['epochs'] = 30
PARAMS['batch_size_trn'] = 128
# PARAMS['dropout'] = [0, 0.3, 0.5, 0.7]   # dropout 0 means we keep all the units
PARAMS['dropout'] = [0, 0.4]
# PARAMS['learn_rate'] = [1e-3, 1e-4, 1e-5]     # 0.0001 usually gives the best results with Adam
PARAMS['learn_rate'] = [1e-4]
PARAMS['patience'] = 0     # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy)

PARAMS['augment_trn'] = False   # augment training set with artificial samples (rotations, etc.). Currently implemented as pre-processing step instead of a tf operation when feeding mini-batches.

PARAMS['subsetting'] = True   # subset datasets for testing purposes

# PARAMS['run_tensorboard'] = True    # does not work because we have to call tensorboard from the Anaconda prompt.


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


## AUGMENT TRAINING SET --------------------------------------------------------------------------------------
# To be put in stream in tf graph: https://www.tensorflow.org/tutorials/deep_cnn
# Y_trn = np.array([val for val in Y_trn for _ in (0, 1)])   ## to replicate labels
#
# if PARAMS['augment_trn']:
#
#     def random_augment(image):
#
#         # Randomly flip the image horizontally.
#         distorted_image = tf.image.flip_left_right(image)
#         #
#         # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
#         # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
#         image = tf.image.per_image_standardization(image)
#
#         distorted_image = tf.image.per_image_standardization(distorted_image)
#
#         return tf.stack([image, distorted_image])
#
#
#     graphAugment = tf.Graph()
#     with graphAugment.as_default():
#         with tf.device('/cpu:0'):
#             X_trn_img_2augment_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X_img_2augment_ph')
#             X_trn_img_augmented = tf.map_fn(random_augment, X_trn_img_2augment_ph)
#
#     with tf.Session(graph=graphAugment, config=tf.ConfigProto(log_device_placement=PARAMS['dev_placement'])) as sess:
#
#         sess.run(tf.global_variables_initializer())
#
#         X_trn_img_augmented = sess.run(X_trn_img_augmented, feed_dict={X_trn_img_2augment_ph: X_trn_img})
#
#         #
#         # for i in range(10):
#         #     plt.imshow(X_trn_img[i,:,:,:])
#         #     plt.savefig(os.path.join(fig_dir, "orig_%d.png" % i))
#         #     plt.imshow(X_trn_img_augmented[i,:,:,:])
#         #     plt.savefig(os.path.join(fig_dir, "augmented_%d.png" % i))
#
#         X_trn_img = np.concatenate([X_trn_img_augmented[:, 0, :,  : , :], X_trn_img_augmented[:, 0, :, :, :]], axis=0)


## DEFINE CNN ------------------------------------------------------------------

## Define CNN with 2 conv layers
def cnn_2convs(X_img, n_classes, dropout, reuse, is_training):

    train_summaries_list = []   # list containg the summaries to be merged together later on

    with tf.variable_scope('ConvNet', reuse=reuse):  # set reuse to True when one wants to use same weights (in test)

        ## Convolutional Layer #1
        # Computes 64 features using a 5x5 filter with ReLU activation.
        # Input Tensor Shape: [batch_size, 32, 32, 3]
        # Output Tensor Shape: [batch_size, 32, 32, filters]
        with tf.name_scope('Conv_1'):
            ## With padding="same" we add as many 0 values at the border as needed to preserve the size of the input
            conv1 = tf.layers.conv2d(X_img, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
            train_summaries_list.append(tf.summary.histogram("output_conv1", conv1))  # append the histogram summary to the list of summaries to be passed to tensorboard

        ## Batch normalization by standardization of each training batch
        with tf.name_scope('Batch_Norm_1'):
            bn1 = tf.layers.batch_normalization(conv1, center=True, scale=True, training=is_training)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set
            train_summaries_list.append(tf.summary.histogram("output_bn1", bn1))

        ## Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 32, 32, filters]
        # Output Tensor Shape: [batch_size, 16, 16, filters]
        with tf.name_scope('MaxPool_1'):
            pool1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2, 2], strides=2)
            train_summaries_list.append(tf.summary.histogram("output_pool1", pool1))

        ## Convolutional Layer #2
        with tf.name_scope('Conv_2'):
            conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        with tf.name_scope('Batch_Norm_2'):
            bn2 = tf.layers.batch_normalization(conv2, center=True, scale=True, training=is_training)

        ## Pooling Layer #2
        # Input Tensor Shape: [batch_size, 16, 16, filters]
        # Output Tensor Shape: [batch_size, 8, 8, filters]
        with tf.name_scope('MaxPool_2'):
            pool2 = tf.layers.max_pooling2d(inputs=bn2, pool_size=[2, 2], strides=2)

        ## Fully connected (dense) Layer
        # Input Tensor Shape: [batch_size, 8, 8, filters]
        # Output Tensor Shape: [batch_size, 1024]
        with tf.name_scope('FC_1'):
            pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
            fc1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        ## Dropout layer: will only be performed if training is True
        with tf.name_scope('Dropout'):
            fc1 = tf.layers.dropout(inputs=fc1, rate=dropout, training=is_training)

        ## Final output layer producing a predicted value (logit) per each class
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, n_classes]
        with tf.name_scope('Softmax_output'):
            output = tf.layers.dense(inputs=fc1, units=n_classes)
            ## Because 'softmax_cross_entropy_with_logits' (used to compute loss) already applies softmax, we only apply softmax to testing network
            if not is_training:
                output = tf.nn.softmax(output)

        return output, train_summaries_list      # return the actual output of the CNN (logits in training and probabilities in test) and the list of summaries to plot in tb

## Wrapper function for the model with learning_rate and dropout as hyperparameters to tune by grid-search
def CIFAR_model(learning_rate, dropout, patience):

    hpar_str = "lr%g_do%g" % (learning_rate, dropout)   # string used to define different directories for each model (hyperparameter combination lr, do) both for model and log files

    tf.set_random_seed(2017)

    val_summaries_list = []   # initialize list of summaries for the validation run

    ## BUILD GRAPH ---------------------------------------------------------------------

    ## Definition of the tf graph to be run in the tf session
    graphCNN = tf.Graph()
    with graphCNN.as_default():

        ## Placeholders with input data (X, Y) and batch size to setup the batch iterator
        with tf.name_scope('input_setup_batches_ph'):
            X_img_2prepr_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X_img_2prepr_ph')
            Y_2prepr_ph = tf.placeholder(tf.float32, [None, 10], name='Y_2prepr_ph')
            batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')

        ## Setup the batch iterator with the Dataset API
        with tf.name_scope('setup_batches'):
            dataset = tf.data.Dataset.from_tensor_slices((X_img_2prepr_ph, Y_2prepr_ph))
            dataset = dataset.shuffle(tf.cast(tf.shape(X_img_2prepr_ph)[0], tf.int64))   # tf.shape(X_img_2prepr_ph)[0] is the number of samples of the dataset
            dataset = dataset.repeat()   # to restart over again when end is reached
            dataset = dataset.batch(batch_size_ph)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()  # next_element is the tf op to get the next batch (last op of the graph that will be called in the session)

        ## Placeholders with input data (X, Y) for network training and inference
        with tf.name_scope('input_ph'):
            X_img_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X_img_ph')
            Y_ph = tf.placeholder(tf.float32, [None, 10], name='Y_ph')

        ## Create a graph for training
        with tf.name_scope('train_CNN'):
            logits_train, train_summaries_list = cnn_2convs(X_img_ph, PARAMS['nr_classes'], dropout, reuse=False, is_training=True)   # outputs train logits and list of training summaries (otherwise they would not be visible bc in the function scope)
            logits_train_summary = tf.summary.histogram('logits_train', logits_train)  # get and add histogram of train logits to training summaries
            train_summaries_list.append(logits_train_summary)

        ## Create another graph for testing that reuse the same weights, but has different behavior for 'dropout' (not applied).
        with tf.name_scope('test_CNN'):
            with tf.device("/cpu:0"):   # to avoid memory problems with the already loaded GPU
                prob_test, _ = cnn_2convs(X_img_ph, PARAMS['nr_classes'], dropout, reuse=True, is_training=False)
                prob_test_summary = tf.summary.histogram('prob_test', prob_test)  # to see decaying learning rate
                val_summaries_list.append(prob_test_summary)

        ## Define loss and optimizer (with train logits, for dropout to take effect)
        with tf.name_scope('loss'):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y_ph))

        with tf.name_scope('training'):
            ## Define update_ops to allow batch normalization to update population mean and variance (moving average) in the training op
            ## (to be used to normalize the test set)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimzer.minimize(loss_op)
                learning_rate = tf.summary.scalar('learning_rate', optimzer._lr)  # to see decaying learning rate
                train_summaries_list.append(learning_rate)

        ## Evaluate model (with test logits, for dropout to be disabled)
        with tf.name_scope('accuracy_assessment'):
            ## Separete operations to check for correct predictions and accuracy in train step and test step to allow plotting separately in tb
            correct_pred_train = tf.equal(tf.argmax(logits_train, 1), tf.argmax(Y_ph, 1))
            correct_pred_test = tf.equal(tf.argmax(prob_test, 1), tf.argmax(Y_ph, 1))
            accuracy_op_train = tf.reduce_mean(tf.cast(correct_pred_train, tf.float32))
            accuracy_op_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
            val_accuracy_summary = tf.summary.scalar('val_accuracy', accuracy_op_test)
            val_summaries_list.append(val_accuracy_summary)

            ## Add placeholder to train summary to be able to save average training accuracy for a given epoch (computed in numpy outside of tf session)
            mean_trn_accuracy_ph = tf.placeholder(tf.float32, shape=())
            trn_accuracy_summary = tf.summary.scalar('train_accuracy', mean_trn_accuracy_ph)

        tf.summary.image('input_image', X_img_ph, 3)    # image summary allowing to see input images in tb


    ## RUN SESSION ---------------------------------------------------------------------

    with tf.Session(graph=graphCNN, config=tf.ConfigProto(log_device_placement=PARAMS['dev_placement'])) as sess:

        saver = tf.train.Saver()

        tf.logging.set_verbosity(tf.logging.INFO)   # verbosity of output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        ## Initialize variables (mandatory step) and batch iterator (on trn set)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={X_img_2prepr_ph: X_trn_img, Y_2prepr_ph: Y_trn, batch_size_ph: PARAMS['batch_size_trn']})

        ## Write summaries separately for training and validation to have separate plots in tb
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, hpar_str, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(log_dir, hpar_str, 'val'))
        train_summary_op = tf.summary.merge(train_summaries_list)  # merge only summaries in each list (as opposed to merge_all())
        val_summary_op = tf.summary.merge(val_summaries_list)

        # Training cycle
        best_val_OA = 0
        best_epoch = 0
        print("Start training...")
        for epoch in range(PARAMS['epochs']):
            start_epoch = tic()
            batch_steps = np.ceil(nr_samples_trn / PARAMS['batch_size_trn']).astype(np.int32)  # maximum number of steps in an epoch (last one will be with a smaller batch-size)
            trn_OA_list = []  # store trn OA and loss for averaging over the epoch
            loss_list = []
            for step in range(batch_steps):

                batch_X_img, batch_Y = sess.run(next_element)  # next_element op gives next batch

                ## Main op with training step of the network
                _, trn_OA, loss = sess.run([train_op, accuracy_op_train, loss_op], feed_dict={X_img_ph: batch_X_img, Y_ph: batch_Y})

                ## Include train summaries (2 times slower though)
                # _, trn_OA, loss, summary = sess.run([train_op, accuracy_op_train, loss_op, train_summary_op], feed_dict={X_img_ph: batch_X_img, Y_ph: batch_Y})
                # train_writer.add_summary(summary, epoch)

                trn_OA_list.append(trn_OA)   # grow lists
                loss_list.append(loss)

                ## Assess model at the end of the epoch
                if step == batch_steps-1:
                    val_OA, summary = sess.run([accuracy_op_test, val_summary_op], feed_dict={X_img_ph: X_val_img, Y_ph: Y_val})
                    val_writer.add_summary(summary, epoch)

                    ## Update best val OA only if current one exceeds previous best value
                    if val_OA > best_val_OA:
                        best_val_OA = val_OA
                        best_epoch = epoch
                        epochs_no_improv = 0  # reset value to compare with patience
                        best_model_dir = os.path.join(model_dir, 'best_model_%s' % hpar_str)   # save model with checkpoint in specific hyperpar folder
                        saver.save(sess, os.path.join(best_model_dir, 'model.ckpt'), global_step=epoch)
                    else:
                        epochs_no_improv += 1

            ## Compute epoch means
            mean_trn_OA = np.mean(trn_OA_list)
            mean_loss = np.mean(loss_list)
            print("Epoch %g: mean loss = %g, trn OA = %.2f, val OA = %.2f, %s" % (epoch, mean_loss, mean_trn_OA*100, val_OA*100, toc(start_epoch)))

            summary = sess.run(trn_accuracy_summary, feed_dict={mean_trn_accuracy_ph: mean_trn_OA})
            train_writer.add_summary(summary, epoch)

            ## Earlystopping with a given patience value
            if epochs_no_improv > patience:
                print("Patience reached: earlystopping with best val OA = %.2f (epoch %g)" % (best_val_OA*100, best_epoch))
                break

        train_writer.close()
        val_writer.close()

        return accuracy_op_test, X_img_ph, Y_ph, saver, graphCNN, best_val_OA

## Gridsearch over the hyperparameters
grid_search_list = []   # list to be converted to pd dataframe
for lr in PARAMS['learn_rate']:
    for do in PARAMS['dropout']:
        print('Learning rate = %g, dropout = %g' % (lr, do))

        ## The CIFAR_model() function needs to ouput all the ops and placeholders to be resued in test
        accuracy_op_test, X_img_ph, Y_ph, saver, graphCNN, val_OA = CIFAR_model(learning_rate=lr, dropout=do, patience=PARAMS['patience'])
        grid_search_list.append({'learn_rate': lr, 'dropout': do, 'val_OA': val_OA})  # fill row entries with dictionary

## Get best values
grid_search_df = pd.DataFrame(grid_search_list)   # convert to pd dataframe
grid_search_df.sort_values(by='val_OA', ascending=False, inplace=True)
best_lr = grid_search_df.loc[0, 'learn_rate']
best_do = grid_search_df.loc[0, 'dropout']

## Prediction on test set
with tf.Session(graph=graphCNN) as sess:
    hpar_str = "lr%g_do%g" % (best_lr, best_do)
    best_model_dir = os.path.join(model_dir, 'best_model_%s' % hpar_str)  # points to folder with best model for this combination of hyperparameters
    ckpt = tf.train.get_checkpoint_state(os.path.join(best_model_dir))  # load model from checkpoint with best validation accuracy
    saver.restore(sess, ckpt.model_checkpoint_path)

    test_OA = sess.run(accuracy_op_test, feed_dict={X_img_ph: X_tst_img, Y_ph: Y_tst})  # session for prediction

    val_OA_CHECK = sess.run(accuracy_op_test, feed_dict={X_img_ph: X_val_img, Y_ph: Y_val})  # check to see if we obtain same val OA as in validation


print('Total ' + toc(start_time))


















# if os.path.exists(best_model_dir):
#     shutil.rmtree(best_model_dir)
# builder = tf.saved_model.builder.SavedModelBuilder(best_model_dir)
# builder.add_meta_graph_and_variables(sess, tags='best_model_%s' % hpar_str)

# saver = tf.train.import_meta_graph(os.path.join(model_dir, 'best_model_%s' % hpar_str))
# tf.saved_model.loader.load(sess,
#                            tags='best_model_%s' % hpar_str,
#                            export_dir=os.path.join(model_dir, 'best_model_%s' % hpar_str))


# builder.save()

# graph = tf.get_default_graph()
# accuracy = graph.get_tensor_by_name("accuracy_tensor:0")
# model = graph.get_tensor_by_name("finalnode:0")





# if PARAMS['run_tensorboard']:
#     cmd = 'tensorboard --logdir=%s' % log_dir
#     os.system(cmd)

# loss, _, logits_train_TOCHECK = sess.run(
#     [loss_op, train_op, logits_train],
#     feed_dict={X_img_ph: batch_X_img, Y_ph: batch_Y})

# prob_val_TOCHECK, correct_pred_val_TOCHECK = sess.run([prob_test, correct_pred], feed_dict={X_img_ph: X_val_img, Y_ph: Y_val})
# prob_trn_TOCHECK, correct_pred_trn_TOCHECK = sess.run([prob_test, correct_pred], feed_dict={X_img_ph: batch_X_img, Y_ph: batch_Y})

# n_batches = nr_samples_val // PARAMS['batch_size_val_tst']
# cumulative_accuracy = 0.0
# for _ in range(n_batches):
#     batch_X_img_val, batch_Y_val = sess.run(next_element,
#                                     feed_dict={X_img_2prepr_ph: X_val_img, Y_2prepr_ph: Y_val,
#                                                batch_size_ph: PARAMS['batch_size_val_tst']})
#
#     cumulative_accuracy += sess.run(accuracy, feed_dict={X_img_ph: batch_X_img_val, Y_ph: batch_Y_val})
# OA = cumulative_accuracy / n_batches


## Decaying learning rate
# num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
# decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
#
# # Decay the learning rate exponentially based on the number of steps.
# lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                 global_step,
#                                 decay_steps,
#                                 LEARNING_RATE_DECAY_FACTOR,
#                                 staircase=True)
# tf.summary.scalar('learning_rate', lr)



# plt.imshow(X_trn_img[2, :, :, :])

# ## Open session
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     for epoch in range(10):
#         epoch_loss = 0.0
#         batch_steps = mnist.train.num_examples / batch_size
#         for step in range(batch_steps):
#             batch_x, batch_y = mnist.train.next_batch(batch_size)
#             _, c = session.run([train_op, loss], {x: batch_x, y: batch_y})
#             epoch_loss += c / batch_steps
#         print "Epoch %02d, Loss = %.6f" % (epoch, epoch_loss)


## CREATE, TRAIN AND EVALUATE ESTIMATOR -------------------------------------------------------------------------------------



# ## Create the Estimator
# CNN = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)
#
# ## Set up logging for predictions
# # tensors_to_log = {}
# # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
#
# ## Train the model
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": X_trn_img},
#     y=Y_trn,
#     batch_size=PARAMS['batch_size'],
#     num_epochs=None,   # epoch: one iteration over all of the training data. If set to None the model will train until the specified number of steps is reached
#     shuffle=True)
#
# CNN.train(
#     input_fn=train_input_fn,
#     steps=PARAMS['steps']  # step: every time a batch is fed to the model
#     # hooks=[logging_hook]
# )
#
# ## Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": X_tst_img},
#     y=Y_tst,
#     num_epochs=1,  # means one time over the whole test set
#     shuffle=False)  # go sequentially through the data
# eval_results = CNN.evaluate(input_fn=eval_input_fn)
# print(eval_results)
#
# # writer = tf.summary.FileWriter(log_dir, sess.graph)





