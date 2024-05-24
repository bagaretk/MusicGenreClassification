from tensorflow.keras.datasets import mnist
import pickle as pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from autoencoder import Autoencoder


LEARNING_RATE = 0.0005
BATCH_SIZE = 16
EPOCHS = 70


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(599, 128, 5),
        conv_filters=(32, 64, 64, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1, 2),
        latent_space_dim=8000
    )
    autoencoder.summary()

    autoencoder.compile(learning_rate)
    # Implement early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    autoencoder.train(x_train, batch_size, epochs, callbacks=[early_stopping])
    return autoencoder



if __name__ == "__main__":
    train_size = 800
    # Load data
    data = []

    with open("data.pkl", 'rb') as f:
        content = f.read()
        data = pickle.loads(content)
    # Convert data to NumPy array
    data = np.asarray(data)
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    trainData = data[:train_size]
    x_train = trainData


    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")