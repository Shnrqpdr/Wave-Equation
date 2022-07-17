#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks-per-node=1            #Numero de tarefas por Nó
#SBATCH --ntasks=1                     #Numero total de tarefas MPI
#SBATCH --cpus-per-task=1              #Numero de threads
#SBATCH -p  nvidia_dev                     #Fila (partition) a ser utilizada
#SBATCH -J wave-eq      #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job

#Exibe os nós alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

malha=ZZZZ

#Configura os compiladores
module load cuda/10.1

#exibe informações sobre o executável
cd /scratch/uff21hpc/pedro.cunha/hpc2/lncc/$malha

EXEC=wave.x

./$EXEC > tempo.txt

