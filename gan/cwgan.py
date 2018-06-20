from __future__ import division, print_function

from datetime import datetime

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import initializers
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Lambda, Reshape, multiply)
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model


class CWGAN():
    def __init__(self,
                 input_size,
                 label_size,
                 latent_size,
                 d_layers,
                 optimizer='adam',
                 activation='relu',
                 show_metrics=False,
                 batch_norm=False):

        # Variables
        self.input_size = input_size
        self.label_size = label_size
        self.latent_size = latent_size
        self.layers = d_layers
        self.batch_norm = batch_norm
        self.activation = activation

        optimizer = self.get_optimizer(optimizer)

        # The generator takes noise and the target label (states) as input
        # and generates the corresponding samples of that label
        noise = Input(shape=(self.latent_size, ), name="noise")
        label = Input(shape=(self.label_size, ), name="labels")
        real_samples = Input(shape=(self.input_size,), name="real")

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator([noise, label])

        # First we train the discriminator
        self.generator.trainable = False
        fake_samples = self.generator([noise, label])

        fake = self.discriminator([fake_samples, label])
        valid = self.discriminator([real_samples, label])

        interpolated = Lambda(self.random_weighted_average)([real_samples, fake_samples])
        valid_interp = self.discriminator([interpolated, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.d_model = Model([real_samples, noise, label], [valid, fake, valid_interp])
        self.d_model.compile(
            loss=[self.wasserstein_loss, self.wasserstein_loss, self.gp_loss(interpolated)],
            optimizer=optimizer)

        # Time to train the generator
        self.discriminator.trainable = False
        self.generator.trainable = True

        noise_gen = Input(shape=(self.latent_size,), name="noise_gen")

        fake_samples = self.generator([noise_gen, label])
        valid = self.discriminator([fake_samples, label])

        self.g_model = Model([noise_gen, label], valid)
        self.g_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # self.init_tensorboard()

        # summary = tf.Summary()
        # self.callback.writer.add_summary(summary, 1)
        # self.callback.writer.flush()

        return None

    def build_generator(self, noise_label):
        noise = noise_label[0]
        label = noise_label[1]

        # noise_dense = Dense(self.latent_size)(noise)
        # noise_dense = self.get_activation()(noise_dense)
        # label_dense = Dense(self.label_size)(label)
        # label_dense = self.get_activation()(label_dense)
        x = concatenate([noise, label])

        # x = Dense(self.label_size+self.latent_size)(concat)

        for l in self.layers[::-1]:
            x = Dense(l)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = self.get_activation('relu')(x)

        out = Dense(self.input_size, activation='tanh')(x)
        generator = Model([noise, label], out)

        return generator

    def build_discriminator(self):
        model = Sequential()
        nodes = int(np.median(self.layers))
        activation = self.get_activation('leaky')
        model.add(Dense(nodes, input_dim=(self.input_size + self.label_size)))
        model.add(activation)
        if len(self.layers) > 1:
            for l in self.layers[1:]:
                model.add(Dense(nodes))
                model.add(activation)

        model.add(Dense(1, activation='linear'))
        model.summary()

        data = Input(shape=(self.input_size, ), name="input_data")
        label = Input(shape=(self.label_size, ), name="input_labels")
        model_input = concatenate([data, label])
        validity = model(model_input)
        discriminator = Model([data, label], validity)
        return discriminator

    def get_activation(self, activation='relu'):
        if activation == 'leaky':
            return LeakyReLU(alpha=0.2)
        if activation == 'prelu':
            return PReLU()
        # default to relu
        return Activation('relu')

    def get_optimizer(self, optimizer):
        if optimizer == 'rmsprop':
            return RMSprop(0.0003)

        return Adam(0.0001, 0.0, 0.9)

    def load_model(self, model):
        return load_model(model)

    def random_weighted_average(self, inputs):
        generated, real = inputs
        alpha = K.random_uniform(shape=K.shape(real))
        diff = generated - real
        return real + alpha * diff

    def gp_loss(self, averaged_samples, lambda_weight=10):
        def loss_func(y_true, y_pred):
            gradients = K.gradients(y_pred, averaged_samples)[0]
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(
                gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = lambda_weight * K.square(1 - gradient_l2_norm)
            return K.mean(gradient_penalty)

        return loss_func

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def init_tensorboard(self):
        # We store the runs in subdirectories named by the time
        dirname = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.callback = TensorBoard(log_dir="./logs/{}/".format(dirname))
        self.callback.set_model(self.d_model)
        return None


if __name__ == '__main__':
    cgan = CWGAN(4, 4, 2, [128, 64])
    # plot_model(cgan.generator, show_shapes=True)
    # cgan.train(epochs=2000, batch_size=32, sample_interval=200)
