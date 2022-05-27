#!/bin/bash
#module load gcc/8.3
#module load gnu-common/1.8
#module load cfitsio/3.450_gnu
#module load ccfits/2.5_gnu

module load openmpi/icc/4.0.4
module load cfitsio/3.450_intel

CUSER="pedro.cunha"
filepath=$(pwd)

function wait(){
    while [ $(squeue -u $CUSER | wc -l) -ge 2 ]
    do
            sleep 5
    done
}

echo "verificando tarefa pendente"
wait
echo "verificada"

nodes="1 4 3 2"
cores="1 2 4 6 8 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48"
max=48

for nd in $nodes;
do
	tnode=$((max / nd))

	sed -i "s/#SBATCH --nodes=.*/#SBATCH --nodes=$nd/g" scriptSequana.sh
        sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=$tnode/g" scriptSequana.sh

	mkdir sequana_${tnode}

	for c in $cores;
   	do
      		rm -fr *.CG *.XV *.DM *.ZM
            	sed -i "s/#SBATCH --ntasks=.*/#SBATCH --ntasks=$c/g" scriptSequana.sh

		sbatch scriptSequana.sh

           	wait
        done
done


#exit
