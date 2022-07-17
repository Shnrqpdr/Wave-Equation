import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from mpl_toolkits.mplot3d import Axes3D

pi = constants.pi

x = np.linspace(0, 500, 1000)
y = np.linspace(0, 500, 1000)
x, y = np.meshgrid(x, y)
z = np.sin((2*x)*pi/75)

    
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_zlim(-2.5, 2.5)
ax1.set_title('Initial condition for the wave')
ax1.set_xlabel('$x$ coordinate')
ax1.set_ylabel('$y$ coordinate')

ax1.plot_wireframe(x, y, z, rstride=250, cstride=250)
plt.savefig('initialCondition.png', dpi=400)