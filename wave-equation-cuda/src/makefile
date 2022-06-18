CC=nvcc
CFLAGS=
COMPILER_FLAGS=

wave.x: wave-eq.cu
	$(CC) $(COMPILER_FLAGS) -o wave.x wave-eq.cu $(CFLAGS)

.PHONY: clean

clean:
	rm -fr *.x *.o *.dat *~ core
