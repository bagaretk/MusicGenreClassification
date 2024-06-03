import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from autoencoder import Autoencoder
from tensorflow.keras import mixed_precision

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
EPOCHS = 100


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data.astype(np.float32)


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(599, 128, 5),
        conv_filters=(16, 16, 32),
        conv_kernels=(4, 4, 4),
        conv_strides=(2, 2, 2),
        latent_space_dim=8192
    )
    autoencoder.summary()

    autoencoder.compile(learning_rate)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    autoencoder.train(x_train, batch_size, epochs, callbacks=[early_stopping])

    # Clear memory after training
    tf.keras.backend.clear_session()
    return autoencoder


if __name__ == "__main__":
    train_size = 600

    with open("data.pkl", 'rb') as f:
        data = pickle.load(f)

    data = np.asarray(data)
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    normalized_data = normalize_data(data)
    train_data = normalized_data[:train_size]
    x_train = train_data

    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
