#!/bin/bash

for j in 5000 6000;
do
    mkdir $j

    cd $j

    cp ../wave-aux.cu .

    sed "s/XYXY/$j/g" wave-aux.cu > wave.cu

    echo "Iniciando o cálculo do programa com as melhores flags"

    nvcc -use_fast_math -O3 -o wave.x wave.cu

    ( time ./wave.x ) 2> tempoMelhorFlag.out

    rm *.x

    cd .. 
done
