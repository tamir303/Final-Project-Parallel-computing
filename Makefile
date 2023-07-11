build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c createMPIStruct.c -o createMPIStruct.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions.o createMPIStruct.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	

clean:
	rm -f *.o ./mpiCudaOpemMP

par:
	mpiexec -np 2 ./mpiCudaOpemMP input.txt "par"

seq:
	mpiexec -np 1 ./mpiCudaOpemMP input.txt "seq"

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
