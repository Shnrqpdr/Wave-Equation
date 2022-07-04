#!/bin/bash

for j in 896 1792 2688 3584;
do
    mkdir "malha$j"

    cd "malha$j"

    cp ../wave.cu .

    sed "s/XYXY/$j/g" wave.cu > exec.cu

    echo "Iniciando o cálculo do programa sem as flags"

    nvcc -o wave.x exec.cu

    ( time ./wave.x ) 2> wave_noFlags.dat

    echo "Iniciando o cálculo do programa com as flags"

    for i in $(seq 0 3)
        do 
            echo "Usando as flags -O$i"
            nvcc -O$i -o wave_O$i.x exec.cu
            ( time ./wave_O$i.x ) 2> wave_O$i.dat

            echo "Usando as flags -use_fast_math -O$i"
            nvcc -use_fast_math -O$i -o wave_fast_O$i.x exec.cu
            ( time ./wave_fast_O$i.x ) 2> wave_fast_O$i.dat

            echo "Usando as flags -Xptxas -O$i"
            nvcc -Xptxas -O$i, -o wave_G_O$i.x exec.cu
            ( time ./wave_G_O$i.x ) 2> wave_G_O$i.dat

        done

    rm *.x

    cd .. 
done
