CC=gcc
CFLAGS=-lm
COMPILER_FLAGS=

laplace.x: wave-eq.c
	$(CC) $(COMPILER_FLAGS) wave-eq.c -o wave.x $(CFLAGS)

.PHONY: clean

clean:
	rm -fr *.x *.o *.dat *.png *~ core
