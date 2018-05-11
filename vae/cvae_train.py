import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import keras
from keras.datasets import mnist
from keras.optimizers import RMSprop
from cvae import CVAE
from scipy.misc import imsave

# set parameters
batch_size = 100
latent_dim = 2
nr_epochs = 1
layers = [256, 128]
optimizer = RMSprop(lr=1e-3)

# get MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train model
original_dim = x_train.shape[1]
label_dim = 1

vae_obj = CVAE(
    original_dim,
    label_dim,
    layers,
    activation='relu',
    optimizer=optimizer,
    dropout=0.0)

vae = vae_obj.compile()
vae.summary()
vae.fit(
    [x_train, y_train],
    x_train,
    shuffle=True,
    epochs=nr_epochs,
    batch_size=batch_size,
    validation_data=([x_test, y_test], x_test),
    verbose=1)

# this loop prints the one-hot decodings

#for i in range(n_z+n_y):
#	tmp = np.zeros((1,n_z+n_y))
#	tmp[0,i] = 1
#	generated = decoder.predict(tmp)
#	file_name = './img' + str(i) + '.jpg'
#	print(generated)
#	imsave(file_name, generated.reshape((28,28)))
#	sleep(0.5)

# this loop prints a transition through the number line

pic_num = 0
variations = 30 # rate of change; higher is slower

current_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(current_dir,'images')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

print("Generating images...")
for j in range(latent_dim, latent_dim + label_dim - 1):
	for k in range(variations):
		v = np.zeros((1, latent_dim+label_dim))
		v[0, j] = 1 - (k/variations)
		v[0, j+1] = (k/variations)
		generated = vae_obj.decoder.predict(v)
		pic_idx = j - latent_dim + (k/variations)
		file_name = img_dir + '/img{0:.3f}.jpg'.format(pic_idx)
		imsave(file_name, generated.reshape((28,28)))
		pic_num += 1