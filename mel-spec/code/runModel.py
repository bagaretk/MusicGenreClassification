import os
import numpy as np
import pickle
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import librosa
import matplotlib.pyplot as plt

# Path to the model
MODEL_PATH = "model.final"

# Define labels dictionary
labelsDict = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
}


def prepossessingAudio(audioPath):
    print('Preprocessing ' + audioPath)
    featuresArray = []
    SOUND_SAMPLE_LENGTH = 30000
    HAMMING_SIZE = 100
    HAMMING_STRIDE = 40

    for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
        if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:
            y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            featuresArray.append(S)

            if len(featuresArray) == 599:
                break
    return np.array(featuresArray)


def load_model():
    # Recreate the model structure
    x = tf.placeholder(tf.float32, [None, 599, 128, 5])
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        'wc3': tf.Variable(tf.random_normal([4, 4, 73, 35])),
        'wd1': tf.Variable(tf.random_normal([38 * 8 * 35, 8192])),
        'out': tf.Variable(tf.random_normal([8192, 10]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([149]) + 0.01),
        'bc2': tf.Variable(tf.random_normal([73]) + 0.01),
        'bc3': tf.Variable(tf.random_normal([35]) + 0.01),
        'bd1': tf.Variable(tf.random_normal([8192]) + 0.01),
        'out': tf.Variable(tf.random_normal([10]) + 0.01)
    }

    def conv2d(sound, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1], padding='SAME'), b))

    def max_pool(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(_X, _weights, _biases, _dropout):
        _X = tf.reshape(_X, shape=[-1, 599, 128, 5])
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        conv1 = max_pool(conv1, k=4)
        conv1 = tf.nn.dropout(conv1, _dropout)
        conv1 = tf.nn.relu(conv1)
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        conv2 = max_pool(conv2, k=2)
        conv2 = tf.nn.dropout(conv2, _dropout)
        conv2 = tf.nn.relu(conv2)
        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        conv3 = max_pool(conv3, k=2)
        conv3 = tf.nn.dropout(conv3, _dropout)
        conv3 = tf.nn.relu(conv3)
        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        dense1 = tf.nn.dropout(dense1, _dropout)
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out

    pred = conv_net(x, weights, biases, keep_prob)

    # Load the saved model
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, MODEL_PATH)
    return sess, pred, x, keep_prob


def predict_genre(audio_path, sess, pred, x, keep_prob):
    features = prepossessingAudio(audio_path)
    features = features.reshape(1, 599, 128, 5)

    prediction = sess.run(pred, feed_dict={x: features, keep_prob: 1.0})
    probabilities = tf.nn.softmax(prediction[0])
    probabilities = sess.run(probabilities)

    return probabilities


def plot_pie_chart(probabilities, title):
    labels = [labelsDict[i] for i in range(10)]
    plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(title)
    plt.show()


def process_directory(directory_path):
    sess, pred, x, keep_prob = load_model()
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                print(f'Processing file: {file_path}')
                probabilities = predict_genre(file_path, sess, pred, x, keep_prob)
                plot_pie_chart(probabilities, title=filename)


if __name__ == "__main__":
    directory_path = "D:\\Licenta\\git\\MusicGenreClassification\\mel-spec\\code\\testMusic"
    process_directory(directory_path)
