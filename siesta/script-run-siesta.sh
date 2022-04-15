#!/bin/bash

for i in $(seq 1 8)
do
	{ time mpirun -np $i ../../../siesta < benzene.fdf ;} 2> tempo_core_$i.txt
	rm *.DM *.XV *.CG *.ZM
done
