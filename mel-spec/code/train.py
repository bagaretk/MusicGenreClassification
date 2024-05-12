#!/usr/bin/python

from __future__ import print_function

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle

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
    batch_size = 64
    display_step = 1
    train_size = 800

    # Network Parameters
    # n_input = 599 * 128
    n_input = 599 * 128 * 5
    n_classes = 10
    dropout = 0.75  # Dropout, probability to keep units

    # Load data
    data = []

    with open("data.pkl", 'rb') as f:
        content = f.read()
        data = pickle.loads(content)
    # Convert data to NumPy array
    data = np.asarray(data)

    # Reshape the data
    data_flattened = data.reshape((data.shape[0], -1))
    print("Shape of data after flattening:", data_flattened.shape)

    labels = []
    with open("labels.pkl", 'rb') as f:
        content = f.read()
        labels = pickle.loads(content)
    labels = np.asarray(labels)

    # #Hack
    # data = np.random.random((1000, n_input))
    # labels = np.random.random((1000, 10))

    # Shuffle data
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    labels = labels[permutation]

    # Split Train/Test
    trainData = data[:train_size]
    trainLabels = labels[:train_size]

    testData = data[train_size:]
    testLabels = labels[train_size:]


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 599, 128, 5])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Create model
    def conv2d(sound, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))


    def max_pool(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, 599, 128, 5])

        # Convolution Layer
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = max_pool(conv1, k=4)
        # Apply Dropout
        conv1 = tf.nn.dropout(conv1, _dropout)
        # Apply activation function
        conv1 = tf.nn.relu(conv1)

        # Convolution Layer
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = max_pool(conv2, k=2)
        # Apply Dropout
        conv2 = tf.nn.dropout(conv2, _dropout)
        # Apply activation function
        conv2 = tf.nn.relu(conv2)

        # Convolution Layer
        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = max_pool(conv3, k=2)
        # Apply Dropout
        conv3 = tf.nn.dropout(conv3, _dropout)
        # Apply activation function
        conv3 = tf.nn.relu(conv3)

        # Fully connected layer
        # Calculate the size of the flattened output
        dense1_input_size = conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2] * \
                            conv3.get_shape().as_list()[3]
        print("dense1_input_size =",dense1_input_size)
        # Reshape conv3 output to fit dense layer input
        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        # Relu activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        # Apply Dropout
        dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out


    # Store layers weight & bias\
    weights = {
        # 4x4 conv, 5 inputs, 149 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        # 4x4 conv, 149 inputs, 73 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        # 4x4 conv, 73 inputs, 35 outputs
        'wc3': tf.Variable(tf.random_normal([4, 4, 73, 35])),
        # fully connected, 10640 inputs, 8192 outputs
        'wd1': tf.Variable(tf.random_normal([38 * 8 * 35, 8192])),
        # 8192 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([8192, 10]))  # Adjusted output size to 10
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([149]) + 0.01),
        'bc2': tf.Variable(tf.random_normal([73]) + 0.01),
        'bc3': tf.Variable(tf.random_normal([35]) + 0.01),
        'bd1': tf.Variable(tf.random_normal([8192]) + 0.01),
        'out': tf.Variable(tf.random_normal([n_classes]) + 0.01)  # n_classes = 10
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = getBatch(trainData, trainLabels, batch_size, step)
            # Fit training using batch data
            print("batch_xs.shape=",batch_xs.shape)
            print("batch_ys.shape=", batch_ys.shape)
            #batch_xs_flattened = np.reshape(batch_xs, (batch_size, -1))
            #print("batch_xs_flattened.shape=", batch_xs_flattened.shape)
            #batch_ys_resized = np.reshape(batch_ys, (batch_size, 10))
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        save_path = saver.save(sess, "model.final")
        print("Model saved in file: %s" % save_path)

        # Calculate accuracy
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData,
                                                                 y: testLabels,
                                                                 keep_prob: 1.}))
