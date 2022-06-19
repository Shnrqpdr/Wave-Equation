#!/bin/bash

for j in 1000 2000 3000 4000 5000;
do
    mkdir $j

    cd $j

    cp ../wave-model.cu .

    sed "s/XYXY/$j/g" wave-model.cu > wave.cu

    echo "Iniciando o cálculo do programa sem as flags"

    nvcc -o wave.x wave.cu

    ( time ./wave.x ) 2> noFlags.out

    echo "Iniciando o cálculo do programa com as flags"

    for i in $(seq 0 3)
        do 
            nvcc -O$i -o wave_O$i.x wave.cu
            ( time ./wave_O$i.x ) 2> wave_O$i.out

            nvcc -use_fast_math -O$i -o wave_fast_O$i.x wave.cu
            ( time ./wave_fast_O$i.x ) 2> wave_fast_O$i.out

            nvcc -Xptxas -O$i, -o wave_G_O$i.x wave.cu
            ( time ./wave_G_O$i.x ) 2> wave_G_O$i.out

        done

    cd .. 
done