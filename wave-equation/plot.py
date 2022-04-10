import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# %matplotlib inline

coluna=pd.read_csv("WaveStatic.dat", header = 0, sep='\s+')

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('rainbow')
ax1.set_title('Wave function at final instant of time')
ax1.set_xlabel('$x$ coordinate')
ax1.set_ylabel('$y$ coordinate')

surf1 = ax1.plot_trisurf(coluna['x'], coluna['y'], coluna['f'], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

#plt.show()
plt.savefig('plotWave.png', dpi=400)