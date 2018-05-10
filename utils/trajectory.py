import numpy as np

class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1

    def getPointsOnSurface(self, n):
        # generate vector of random angles
        angles = np.random.uniform(-np.pi, np.pi, n)

        # generate matrix of x and y coordinates
        x = np.cos(angles)
        y = np.sin(angles)
        return angles, x, y
