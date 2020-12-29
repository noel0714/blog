import matplotlib.pyplot as plt
import numpy as np
from numba import jit


@jit(nopython=True)
def getGaussian(x, y, omega=1.5):
    z = [1 / (2 * np.pi * (omega**2)) * np.exp(-(X**2 + Y**2) / (2 * omega**2)) for X in x for Y in y]
    z = np.array(z).reshape((len(x), len(y)))

    return z


x = np.arange(-4, 4, 0.01)
y = np.arange(-4, 4, 0.01)
z = getGaussian(x, y)

x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(x, y, z, color='Blue')
plt.show()