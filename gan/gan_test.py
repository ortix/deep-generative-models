import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def getSamples(n):
    # generate vector of random angles
    angles = np.random.uniform(-np.pi, np.pi, n)

    # generate matrix of x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    return angles, x, y


angles, x,y = getSamples(200)

generator = load_model('gan_trained.h5')
generator.summary()

# Generate samples from noise
noise = np.random.normal(0, 1, (10000, 100))
samples = generator.predict(noise)
samples = np.squeeze(samples, axis=2)

plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.show()
