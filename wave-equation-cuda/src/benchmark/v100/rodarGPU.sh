#!/bin/bash

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

for j in 896 1792 2688 3584;
do
    mkdir $j

    cd $j

    cp ../wave-aux.cu .
    cp ../scriptLNCCsequana.sh .

    sed "s/XYXY/$j/g" wave-aux.cu > wave.cu
    sed "s/ZZZZ/$j/g" scriptLNCCsequana.sh > script.sh

    echo "Iniciando o cálculo do programa "

    nvcc -use_fast_math -O3 -o wave.x wave.cu

    sbatch script.sh

    wait

    cd .. 
done
