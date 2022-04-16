import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import numpy as np
import pandas as pd

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe' # PRO WINDOWS

t = 100

data = pd.read_csv("../Wave.dat", header = 0, sep='\s+')

x = data['x']
y = data['y']
f = data['f']

x = np.split(x, t)
y = np.split(y, t)
f = np.split(f, t)

print('teste x: ', len(x))
print('teste y: ', len(y))
print('teste f: ', len(f))

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(projection='3d')
    
mycmap = plt.get_cmap('rainbow')
ax1.set_xlabel('$x$ coordinate')
ax1.set_ylabel('$y$ coordinate')
#plt.style.use('dark_background')
plt.title(r'Time evolution of wave')

surf1 = ax1.plot_trisurf(x[0], y[0], f[0], cmap=mycmap)

plt.close()
    
def init():
    return 5
    
def anim(i):
    ax1.clear()
    surf1 = ax1.plot_trisurf(x[i], y[i], f[i], cmap=mycmap)
    return surf1

    
anim = FuncAnimation(fig, anim, t, interval=10, blit=False)
anim.save('wave1.mp4', fps=30)