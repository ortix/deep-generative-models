from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, PReLU, Input, Lambda
from keras.optimizers import Adam
import numpy as np
from functools import partial


class CWGAN():

    def __init__(self,
                 input_size,
                 label_size,
                 latent_size,
                 d_layers,
                 activation='relu'):
        self.input_size = input_size
        self.label_size = label_size
        self.latent_size = latent_size
        self.layers = d_layers
        self.activation = activation

        self.generator = self.create_generator()
        self.generator.summary()
        self.discriminator = self.create_discriminator()
        self.discriminator.summary()

        self.g_model = self.compile_generator()
        self.d_model = self.compile_discriminator()

        return None

    def compile_generator(self):
        '''
        First configures the generator and discriminator and then compiles the generator.
        We need to disable the discriminator since we train it separately from the generator.
        We re-enable the discriminator once the generator has been compiled
        '''
        self.discriminator_trainable(False)
        generator_input = Input(shape=(self.latent_size, ))
        generator_layers = self.generator(generator_input)
        discriminator_layers_for_generator = self.discriminator(
            generator_layers)
        generator_model = Model(
            inputs=[generator_input],
            outputs=[discriminator_layers_for_generator])
        # We use the Adam paramaters from Gulrajani et al.
        generator_model.compile(
            optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
            loss=self.wasserstein_loss)
        self.discriminator_trainable(True)
        return generator_model

    def compile_discriminator(self):
        self.generator_trainable(False)

        real_samples = Input(shape=(self.input_size, ))
        generator_input_for_discriminator = Input(shape=(self.latent_size, ))
        generated_samples_for_discriminator = self.generator(
            generator_input_for_discriminator)
        discriminator_output_from_generator = self.discriminator(
            generated_samples_for_discriminator)
        discriminator_output_from_real_samples = self.discriminator(
            real_samples)

        # Generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
        averaged_samples = Lambda(self.random_weighted_average)(
            [generated_samples_for_discriminator, real_samples])
        averaged_samples_out = self.discriminator(averaged_samples)

        discriminator_model = Model(
            inputs=[real_samples, generator_input_for_discriminator],
            outputs=[
                discriminator_output_from_real_samples,
                discriminator_output_from_generator, averaged_samples_out
            ])

        discriminator_model.compile(
            optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
            loss=[
                self.wasserstein_loss, self.wasserstein_loss,
                self.gp_loss(averaged_samples)
            ])
        self.generator_trainable(True)
        return None

    def random_weighted_average(self, inputs):
        generated, real = inputs
        alpha = K.random_uniform(shape=K.shape(real))
        diff = generated - real
        return real + alpha * diff

    def discriminator_trainable(self, trainable):
        for layer in self.discriminator.layers:
            layer.trainable = trainable
        self.discriminator.trainable = trainable
        return None

    def generator_trainable(self, trainable):
        for layer in self.generator.layers:
            layer.trainable = trainable
        self.generator.trainable = trainable
        return None

    def create_generator(self):
        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.latent_size))
        model.add(self.get_activation())
        if len(self.layers) > 1:
            # Reverse layers for generator (small to large)
            for l in self.layers[::-1]:
                model.add(Dense(l))
                model.add(self.get_activation())

        # Even though this is the output of the generator, the data itself is called the input to the model
        model.add(Dense(self.input_size))
        return model

    def create_discriminator(self):
        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_size))
        model.add(self.get_activation())
        if len(self.layers) > 1:
            for l in self.layers:  # self.layers -1
                model.add(Dense(l, kernel_initializer='he_normal'))
                model.add(self.get_activation())

        model.add(Dense(1, kernel_initializer='he_normal'))
        return model

    def get_activation(self):
        if self.activation == 'leaky':
            return LeakyReLU()
        if self.activation == 'prelu':
            return PReLU()
        # default to relu
        return Activation('relu')

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

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


if __name__ == "__main__":
    nn = CWGAN(4, 4, 2, [64, 32])
