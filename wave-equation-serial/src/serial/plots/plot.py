import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from dask import dataframe as dd
# %matplotlib inline

file=dd.read_csv("../WaveStatic.dat", header = 0, sep='\s+')

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('rainbow')
ax1.set_title('Wave function at final instant of time')
ax1.set_xlabel('$x$ coordinate')
ax1.set_ylabel('$y$ coordinate')

surf1 = ax1.plot_trisurf(file['x'], file['y'], file['f'], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

#plt.show()
plt.savefig('plotWave5.png', dpi=400)