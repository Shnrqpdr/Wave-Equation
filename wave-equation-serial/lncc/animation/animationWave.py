import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import numpy as np
import pandas as pd

#plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe' # PRO WINDOWS

t = 300

data = pd.read_csv("Wave.dat", header = 0, sep='\s+')

x = data['x']
y = data['y']
f = data['f']

x = np.array_split(x, t)
y = np.array_split(y, t)
f = np.array_split(f, t)

print('teste x: ', len(x))
print('teste y: ', len(y))
print('teste f: ', len(f))

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(projection='3d')
    
mycmap = plt.get_cmap('rainbow')
ax1.set_xlabel('$x$ coordinate', color='white')
ax1.set_ylabel('$y$ coordinate', color='white')

#ax1.xaxis.label.set_color('white')        #setting up X-axis label color to yellow
#ax1.yaxis.label.set_color('white')          #setting up Y-axis label color to blue

ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
ax1.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black


plt.title(r'Time evolution of wave')
ax1.set_facecolor('#000000')
#plt.style.use('dark_background')
surf1 = ax1.plot_trisurf(x[0], y[0], f[0], cmap=mycmap)

plt.close()
    
def init():
    return 5
    
def anim(i):
    ax1.clear()
    #plt.style.use('dark_background')
    ax1.set_facecolor('#000000')
    surf1 = ax1.plot_trisurf(x[i], y[i], f[i], cmap=mycmap)
    return surf1

anim = FuncAnimation(fig, anim, t, interval=10, blit=False)
    
#plt.style.use('dark_background')
anim.save('wave1.mp4', fps=30, savefig_kwargs={'facecolor':'black'})