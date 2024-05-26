import os
import pickle
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


class SparseRegularization(Layer):
    def __init__(self, beta, rho, **kwargs):
        super(SparseRegularization, self).__init__(**kwargs)
        self.beta = beta
        self.rho = rho

    def call(self, inputs):
        p_hat = K.mean(inputs, axis=0)
        epsilon = 1e-10  # small value to avoid log(0)
        sparsity_loss = tf.reduce_sum(
            self.rho * K.log(self.rho / (p_hat + epsilon)) +
            (1 - self.rho) * K.log((1 - self.rho) / (1 - p_hat + epsilon))
        )
        self.add_loss(self.beta * sparsity_loss)
        return inputs

def sparse_loss(y_true, y_pred, model, lambda_l2):
    mse_loss = MeanSquaredError()(y_true, y_pred)
    l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in model.trainable_weights])
    total_loss = mse_loss + lambda_l2 * l2_loss

    tf.print("mse_loss:", mse_loss)
    tf.print("l2_loss:", l2_loss)
    tf.print("total_loss:", total_loss)

    tf.debugging.check_numerics(mse_loss, "NaN or Inf in mse_loss")
    tf.debugging.check_numerics(l2_loss, "NaN or Inf in l2_loss")
    tf.debugging.check_numerics(total_loss, "NaN or Inf in total_loss")

    return total_loss


class Autoencoder:
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 lambda_l2=0.005,
                 beta=0.1,
                 rho=0.19):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.lambda_l2 = lambda_l2
        self.beta = beta
        self.rho = rho

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=1e-5):
        optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
        self.model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: sparse_loss(y_true, y_pred, self.model, self.lambda_l2))


    def train(self, x_train, batch_size, num_epochs, callbacks=None):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       callbacks=callbacks)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            self.lambda_l2,
            self.beta,
            self.rho
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=5,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        trimmed_output = output_layer[:, :599, :, :]
        return trimmed_output

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(self.lambda_l2),
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output",
                  activity_regularizer=regularizers.L1(self.beta))(x)
        x = SparseRegularization(self.beta, self.rho)(x)
        return x

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(599, 128, 5),
        conv_filters=(8, 16, 32),
        conv_kernels=(4, 4, 4),
        conv_strides=(2, 2, 2,),
        latent_space_dim=8192,
        lambda_l2=0.005,
        beta=0.01,
        rho=0.05
    )
    autoencoder.summary()