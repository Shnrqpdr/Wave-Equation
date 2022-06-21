#!/bin/bash

for j in 896 1792 2688 3584;
do
    mkdir $j

    cd $j

    cp ../wave-aux.cu .

    sed "s/XYXY/$j/g" wave-aux.cu > wave.cu

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

    rm *.x

    cd .. 
done
