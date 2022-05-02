#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=24
#SBATCH -p cpu_dev                      #Fila (partition) a ser utilizada
#SBATCH -J siesta			               #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job
#SBATCH --time=00:20:00

#Exibe os nós alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIRSS

source /scratch/app/modulos/intel-psxe-2019.sh
module load openmpi/icc/2.0.4.2

#Configura o executavel
EXEC=/scratch/uff21hpc/pedro.cunha/Siesta/siesta-v4.1-b4/Obj/siesta

#exibe informações sobre o executável
#/usr/bin/ldd $EXEC

cd /scratch/uff21hpc/pedro.cunha/Siesta/siesta-v4.1-b4/Obj/Tests/benzene/work

{ time srun -n $SLURM_NTASKS $EXEC < benzene.fdf ; } 2> b710_${SLURM_NTASKS_PER_NODE}/tempo${SLURM_NTASKS}.txt

