#!/usr/bin/python

from __future__ import print_function

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
import numpy as np
import pickle
from autoencoder import Autoencoder

def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch], labels[startOfBatch:endOfBatch]
    else:
        dataBatch = data[startOfBatch:endOfBatch]
        labelsBatch = labels[startOfBatch:endOfBatch]
        return dataBatch, labelsBatch

if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 1
    display_step = 1
    train_size = 800

    # Network Parameters
    n_input = 8192
    n_classes = 10
    dropout = 0.75  # Dropout, probability to keep units

    # Load trained autoencoder
    autoencoder = Autoencoder.load("model")

    # Load data
    with open("data.pkl", 'rb') as f:
        data = pickle.load(f)
    data = np.asarray(data)

    # Reshape the data
    # data_flattened = data.reshape((data.shape[0], -1))
    # print("Shape of data after flattening:", data_flattened.shape)

    with open("labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    labels = np.asarray(labels)

    # Shuffle data
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    labels = labels[permutation]

    # Normalize data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    normalized_data = normalized_data.astype(np.float32)
    print(f"type of normalized_data: {normalized_data.dtype}")
    print(f"shape of normalized_data: {normalized_data.shape}")
    
    # Get latent representations from autoencoder
    _, data = autoencoder.reconstruct(normalized_data)
    # latent_representations = latent_representations.numpy() #make sure data is munpy array
    print("Shape of data after autoencoder:", data.shape)

    # Split Train/Test
    trainData = normalized_data[:train_size]
    trainLabels = labels[:train_size]

    testData = normalized_data[train_size:]
    testLabels = labels[train_size:]

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create model
    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, n_input])

        # Fully connected layer 1
        dense1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['wd1']), _biases['bd1']))
        dense1 = tf.nn.dropout(dense1, _dropout)

        # Fully connected layer 2
        dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd2']), _biases['bd2']))
        dense2 = tf.nn.dropout(dense2, _dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
        out = tf.nn.softmax(out)  # Apply softmax to the output
        return out

    # Store layers weight & bias
    weights = {
        'wd1': tf.Variable(tf.random_normal([n_input, 2048])),
        'wd2': tf.Variable(tf.random_normal([2048, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bd1': tf.Variable(tf.random_normal([2048]) + 0.01),
        'bd2': tf.Variable(tf.random_normal([1024]) + 0.01),
        'out': tf.Variable(tf.random_normal([n_classes]) + 0.01)
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_xs, batch_ys = getBatch(trainData, trainLabels, batch_size, step)
            print("batch_xs.shape=", batch_xs.shape)
            print("batch_ys.shape=", batch_ys.shape)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        save_path = saver.save(sess, "model.final")
        print("Model saved in file: %s" % save_path)

        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabels, keep_prob: 1.}))

