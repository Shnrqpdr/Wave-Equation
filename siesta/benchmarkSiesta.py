import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

file1 = pd.read_csv("temposLab107.txt", header = 0, sep='\s+')
file2 = pd.read_csv("temposB710.txt", header = 0, sep='\s+')
file3 = pd.read_csv("temposSequana.txt", header = 0, sep='\s+')

fig1 = plt.figure(figsize=(8,6))

x=np.arange(0,50,2)

axes = fig1.add_axes([0.1,0.1,0.8,0.8])

axes.plot(file1['core'], file1['tempo'], ls='-.', marker='o',markersize=8, label="Lab 107-C")
axes.plot(file2['core'], file2['tempo'], ls='-.', marker='o',markersize=8, label="LNCC - B710")
axes.plot(file3['core'], file3['tempo'], ls='-.', marker='o',markersize=8, label="LNCC - SequanaX")

axes.set_xlabel('Cores')
axes.set_ylabel("Time (s)")

axes.set_title("Comparison of MPI implementation on each case")

plt.grid(linestyle='-', linewidth=0.5)
axes.legend(loc='upper right')

plt.savefig("benchmarkSiestaMPI.pdf",dpi=600)