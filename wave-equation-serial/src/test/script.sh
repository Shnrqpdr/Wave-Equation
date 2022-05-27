#!/bin/bash

flagOX="0 1 2 3"

for flag in $flags;
do
    sed "s/COMPILER_FLAGS=.*/COMPILER_FLAGS=-O$flag/g" teste.txt
done