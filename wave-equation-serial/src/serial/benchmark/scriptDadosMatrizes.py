import numpy as np 
import pandas as pd 

def converterTempo(data):
    if(len(data) == 10):
        minutes = float(data[0:2])
        seconds = float(data[3:9])
        minutes = minutes*60
        
    if(len(data) == 9):
        minutes = float(data[0:1])
        seconds = float(data[2:8])
        minutes = minutes*60
    
    if(len(data) == 8):
        minutes = float(data[0:1])
        seconds = float(data[2:7])
        minutes = minutes*60
    
    return minutes + seconds

cores = list(range(1, 9, 1))

tempos = []
eficiencias = []
speedups = []

for core in cores:
    file = pd.read_csv(f'tempo_core_{core}.dat', header = None, sep='\s+')
    
    if ',' in file[1][0]:
        tempo = file[1][0].replace(',', '.')
    else:
        tempo = file[1][0]
    
    tempo = converterTempo(tempo)
    tempos.append(tempo)
    
for i in range(np.size(cores)):
    eficiencias.append((1 - (tempos[i]/tempos[0]))*100)
    
for i in range(np.size(cores)):
    speedups.append((tempos[0]/tempos[i]))
       
print(tempos)
print(eficiencias)
print(speedups)

tab = '\t'

dic = {"cores": cores,
       "tempo": tempos,
       "eficiencia": eficiencias,
       "speedup": speedups}

dataFrame = pd.DataFrame(dic)
dataFrame.to_csv("dados.dat", index=False, sep=tab)
