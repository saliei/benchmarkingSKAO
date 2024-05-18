all:
	#mpicc -shared -fopenmp -O3 -march=native -o libgrid.so -fPIC libgrid.c -Wall -lm
	nvcc -shared -O3 -arch=sm_70 -o libgrid_cuda.so -Xcompiler -fPIC libgrid.cu

clean:
	rm libgrid.so

