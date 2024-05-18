all:
	nvcc -shared -O3 -arch=sm_70 -o libgrid_cuda.so -Xcompiler -fPIC libgrid.cu

clean:
	rm libgrid.so

