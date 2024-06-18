import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.callbacks import EarlyStopping
from autoencoder import Autoencoder
from tensorflow.keras import mixed_precision
def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch], labels[startOfBatch:endOfBatch]
    else:
        dataBatch = data[startOfBatch:endOfBatch]
        labelsBatch = labels[startOfBatch:endOfBatch]
        return dataBatch, labelsBatch

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configure environment variable for memory allocator
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

LEARNING_RATE = 0.0005
BATCH_SIZE = 4
EPOCHS = 120


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(599, 128, 5),
        conv_filters=(16, 16, 32),
        conv_kernels=(4, 4, 4),
        conv_strides=(2, 2, 2),
        latent_space_dim=14000
    )
    autoencoder.summary()
    print(f"x_train shape : {x_train.shape}")
    autoencoder.compile(learning_rate)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    autoencoder.train(x_train, batch_size, epochs, callbacks=[early_stopping])
    #reconstruction accuracy of the autoencoder
    reconstructed_data = autoencoder.encoder.predict(x_train)
    if reconstructed_data.shape[-1] == 1:
        reconstructed_data = np.squeeze(reconstructed_data, axis=-1)
    autoencoderAccuracy = (1 - np.mean(x_train - autoencoder.decoder.predict(reconstructed_data)) ** 2) * 100
    print(f"Autoencoder reconstruction accuracy(FITSCORE): {autoencoderAccuracy}")
    # Clear memory after training
    tf.keras.backend.clear_session()
    return autoencoder


if __name__ == "__main__":
    train_size = 1000

    with open("data.pkl", 'rb') as f:
        data = pickle.load(f)

    data = np.asarray(data)
    permutation = np.random.permutation(len(data))
    data = data[permutation]

    train_data = data[:train_size]
    x_train = normalize_data(train_data)

    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("model")
    latent_representations = autoencoder.encoder.predict(data)
    tf.disable_v2_behavior()
    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 64
    display_step = 1
    train_size = 800

    # Network Parameters
    n_input = 14000
    n_classes = 10
    dropout = 0.75  # Dropout, probability to keep units

    with open("labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    labels = np.asarray(labels)

    # Shuffle labels
    labels = labels[permutation]

    #latent_representations = latent_representations.numpy()  # Ensure the result is a NumPy array
    print("Shape of data after autoencoder:", latent_representations.shape)

    # Split Train/Test
    trainData = latent_representations[:train_size]
    trainLabels = labels[:train_size]

    testData = latent_representations[train_size:]
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