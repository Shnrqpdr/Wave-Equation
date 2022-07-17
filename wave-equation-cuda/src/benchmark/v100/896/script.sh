#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks-per-node=1            #Numero de tarefas por Nó
#SBATCH --ntasks=1                     #Numero total de tarefas MPI
#SBATCH --cpus-per-task=1              #Numero de threads
#SBATCH -p  sequana_gpu_dev                     #Fila (partition) a ser utilizada
#SBATCH -J wave-eq      #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job
#SBATCH --time=00:20:00


#Exibe os nós alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

malha=896

module load sequana/current

#Configura os compiladores
module load cuda/10.1_sequana

#exibe informações sobre o executável
cd /scratch/uff21hpc/pedro.cunha/hpc2/lncc/$malha

EXEC=wave.x

./$EXEC > tempo.txt

