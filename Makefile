build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c createMPIStruct.c -o createMPIStruct.o
	mpicxx -fopenmp -c general.c -o general.o
	mpicxx -fopenmp -c parallelFunctions.c -o parallelFunctions.o
	mpicxx -fopenmp -c sequentialFunctions.c -o sequentialFunctions.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o general.o cudaFunctions.o createMPIStruct.o parallelFunctions.o sequentialFunctions.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	
clean:
	rm -f *.o ./mpiCudaOpemMP

par:
	mpiexec -np 3 ./mpiCudaOpemMP input.txt "par"

seq:
	mpiexec -np 1 ./mpiCudaOpemMP input.txt "seq"

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP "par"
