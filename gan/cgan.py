from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import initializers
import tensorflow as tf
import keras.backend as K
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np


class CGAN():
    def __init__(self,
                 input_size,
                 label_size,
                 latent_size,
                 d_layers,
                 optimizer='adam',
                 activation='leaky',
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

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Get a trainable generator and runtime (rt) generator
        self.generator = self.build_generator([noise, label])

        generated_samples = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([generated_samples, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(
            loss=['binary_crossentropy'], optimizer=optimizer)

        self.init_tensorboard()

        summary = tf.Summary()
        self.callback.writer.add_summary(summary, 1)
        self.callback.writer.flush()

        return None

    def build_generator(self, noise_label):
        noise = noise_label[0]
        label = noise_label[1]

        noise_dense = Dense(self.latent_size)(noise)
        noise_dense = self.get_activation()(noise_dense)
        label_dense = Dense(self.label_size)(label)
        label_dense = self.get_activation()(label_dense)
        x = concatenate([noise_dense, label_dense])

        # x = Dense(self.label_size+self.latent_size)(concat)

        for l in self.layers[::-1]:
            x = Dense(l)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = self.get_activation()(x)

        out = Dense(self.input_size)(x)
        generator = Model([noise, label], out)

        return generator

    def build_generator_(self):

        activation = self.get_activation()
        model = Sequential()

        model.add(Dense(min(self.layers),
                        input_shape=(self.latent_size+self.label_size,),
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(activation)

        if len(self.layers) > 1:
            for l in self.layers[::-1][1:]:  # reverse layers
                model.add(Dense(l))
                model.add(BatchNormalization(momentum=0.8))
                model.add(activation)

        # Final generator output
        model.add(Dense(self.input_size, activation='tanh'))
        model.summary()

        # Set up inputs
        noise = Input(shape=(self.latent_size,))
        label = Input(shape=(self.label_size,))
        noise_label = Input(shape=(self.label_size+self.latent_size,))
        model_input = concatenate([noise, label])

        # Build models (training and runtime) with inputs and corresponding outputs
        output = model(model_input)
        output2 = model(noise_label)
        generator = Model([noise, label], output, name="generator")
        generator_rt = Model(noise_label, output2, name="generator_rt")
        return generator, generator_rt

    def build_discriminator(self):
        model = Sequential()
        nodes = int(np.median(self.layers))
        activation = self.get_activation()
        model.add(Dense(nodes, input_dim=(self.input_size + self.label_size),
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(activation)
        model.add(Dropout(0.4))
        if len(self.layers) > 1:
            for l in self.layers[1:]:
                model.add(Dense(nodes))
                model.add(activation)
                model.add(Dropout(0.4))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=(self.input_size, ), name="input_data")
        label = Input(shape=(self.label_size, ), name="input_labels")
        model_input = concatenate([data, label])
        validity = model(model_input)
        discriminator = Model([data, label], validity)
        return discriminator

    def get_activation(self):
        if self.activation == 'leaky':
            return LeakyReLU(alpha=0.2)
        if self.activation == 'prelu':
            return PReLU()
        # default to relu
        return Activation('relu')

    def get_optimizer(self, optimizer):
        if optimizer == 'rmsprop':
            return RMSprop(0.0003)

        return Adam(0.0002, 0.5)

    def load_model(self, model):
        return load_model(model)

    def init_tensorboard(self):
        # We store the runs in subdirectories named by the time
        dirname = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.callback = TensorBoard(log_dir="./logs/{}/".format(dirname))
        self.callback.set_model(self.combined)
        return None


if __name__ == '__main__':
    cgan = CGAN(4, 4, 2, [128, 64])
    # plot_model(cgan.generator, show_shapes=True)
    # cgan.train(epochs=2000, batch_size=32, sample_interval=200)
