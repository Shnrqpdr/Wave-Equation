#!/bin/bash

for j in 896 1792 2688 3584;
do
    mkdir $j

    cd $j

    cp ../wave-aux.c .

    sed "s/XXXX/$j/g" wave-aux.c > wave.c

    echo "Iniciando o cálculo do programa sem as flags"

    gcc -o wave.x wave.c

    ( time ./wave.x ) 2> serial.out

    echo "Iniciando o cálculo do programa com as flags"

    for i in $(seq 0 3)
        do 
            gcc -O$i -o wave_O$i.x wave.c
            ( time ./wave_O$i.x ) 2> wave_O$i.out

            gcc -fexpensive-optimizations -m64 -foptimize-register-move -funroll-loops -ffast-math -mtune=native -march=native -O$i -o wave_native_O$i.x wave.c
            ( time ./wave_native_O$i.x ) 2> wave_native_O$i.out

        done

    rm *.x

    cd .. 
done
