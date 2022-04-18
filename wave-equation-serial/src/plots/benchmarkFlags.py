import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

file = pd.read_csv("tempoFlag.txt", header = 0, sep='\s+')

fig = plt.figure(figsize=(8,6))

x=np.arange(0,4,1)

axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(file['flag'], file['t'], ls='-.', marker='o',markersize=8, label="SERIAL Program time")

axes.set_xlabel('Flags O(X)')
axes.set_ylabel("Time (s)")

axes.set_title("Comparison of optimatizations for each flag")

plt.grid(linestyle='-', linewidth=0.5)
axes.legend(loc='upper right')

plt.savefig("benchmarkFlagsSERIAL.pdf",dpi=600)