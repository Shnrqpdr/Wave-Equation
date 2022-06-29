CC=gcc
CFLAGS=-lm
COMPILER_FLAGS=-pg -g -O2 -fexpensive-optimizations -m64 -foptimize-register-move -funroll-loops -ffast-math -mtune=native -march=native

wave.x: wave-lncc.c
	$(CC) $(COMPILER_FLAGS) wave-lncc.c -o wave.x $(CFLAGS)

.PHONY: clean

clean:
	rm -fr *.x *.o *.dat *~ core
