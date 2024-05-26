from tensorflow.keras.datasets import mnist
import pickle as pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from autoencoder import Autoencoder, sparse_loss

LEARNING_RATE = 0.0005
BATCH_SIZE = 8
EPOCHS = 40


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
        conv_filters=(8, 16, 32),
        conv_kernels=(4, 4, 4),
        conv_strides=(2, 2, 2),
        latent_space_dim=8192,
        lambda_l2=0.005,
        beta=0.01,
        rho=0.05
    )
    autoencoder.summary()

    autoencoder.compile(learning_rate)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (x_batch) in enumerate(tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)):
            with tf.GradientTape() as tape:
                y_pred = autoencoder.model(x_batch, training=True)
                loss = sparse_loss(x_batch, y_pred, autoencoder.model, autoencoder.lambda_l2)

            gradients = tape.gradient(loss, autoencoder.model.trainable_variables)
            autoencoder.model.optimizer.apply_gradients(zip(gradients, autoencoder.model.trainable_variables))

            if tf.reduce_any(tf.math.is_nan(loss)):
                print(f"NaN detected in loss at epoch {epoch} step {step}")
                break
            print(f"Step no.:{step}, loss:{loss}")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    return autoencoder



if __name__ == "__main__":
    # Load data
    with open("data.pkl", 'rb') as f:
        data = pickle.load(f)

    # Normalize data
    data = np.asarray(data)
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    trainData = data[:800]
    x_train = (trainData - np.mean(trainData)) / np.std(trainData)

    # Train the autoencoder
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")