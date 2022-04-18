import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from dask import dataframe as dd
# %matplotlib inline

# chunks = pd.read_csv("WaveStatic.dat", header = 0, sep='\s+', chunksize=124900, engine='python')

# file = pd.concat(chunks)

dask_df = dd.read_csv('WaveStatic.dat', header = 0, sep='\s+')

print('Montando a figura\n')

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('rainbow')
ax1.set_title('Wave function at final instant of time')
ax1.set_xlabel('$x$ coordinate')
ax1.set_ylabel('$y$ coordinate')

print('Plotando o gráfico\n')

surf1 = ax1.plot_trisurf(dask_df['x'], dask_df['y'], dask_df['f'], cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

print('Salvando a figura\n')

#plt.show()
plt.savefig('plotWave3.pdf')

print('Fechando a figura\n')
plt.close(fig)