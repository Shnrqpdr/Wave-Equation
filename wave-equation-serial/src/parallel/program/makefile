CC=mpicc
CFLAGS=-lm
COMPILER_FLAGS=

wave.x: wave-eq.c
	$(CC) $(COMPILER_FLAGS) wave-eq.c -o wave.x $(CFLAGS)

.PHONY: clean

clean:
	rm -fr *.x *.o *.dat *~ core