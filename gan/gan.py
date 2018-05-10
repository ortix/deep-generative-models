from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

import sys

import numpy as np


class GAN():
    def __init__(self):
        self.shape = (2, 1)

        optimizer = RMSprop(0.002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100, ))
        points = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(points)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100, )

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.1))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.1))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.1))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Reshape(self.shape))

        model.summary()
        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        shape = self.shape

        model = Sequential()

        model.add(Flatten(input_shape=shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1, activation='sigmoid'))

        print('Discriminator:')
        model.summary()

        points = Input(shape=shape)
        validity = model(points)

        return Model(points, validity)

    # Sample unit circle
    def getSamples(self, n):
        # generate vector of random angles
        angles = np.random.uniform(0, 2*np.pi, n)

        # generate matrix of x and y coordinates
        x = np.cos(angles)
        y = np.sin(angles)
        return angles, x, y

    def train(self, epochs, batch_size=128, save_interval=100):

        # Load the dataset
        n = 10000
        angles, x_pos, y_pos = self.getSamples(n)
        X_train = (np.array([x_pos, y_pos]).T +1) /2
        X_train = np.expand_dims(X_train, axis=3)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            points = X_train[idx]
            # arr = np.linalg.norm(points,axis=1)
            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_points = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                points, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                gen_points, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            if (epoch % 10 == 0):
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        noise = np.random.normal(0, 1, (10000, 100))
        samples = self.generator.predict(noise)
        samples = np.squeeze(samples, axis=2)
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.savefig("./images/gan_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=5000, batch_size=50, save_interval=200)
    gan.generator.save('gan_trained.h5')
