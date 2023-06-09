#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include "mathCalc.h"

__global__  void calcCords(int *cords) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    cords[i] = 
}


int computeOnGPU(Point* points, Cord* cords, int pSize, int cSize) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t p_tSize = pSize * sizeof(Point);
    size_t c_tSize = cSize * sizeof(Cord);
    int thread_num = pSize;
    int block_num = cSize;

    // Allocate memory on GPU to copy the data from the host
    Point *p_A;
    err = cudaMalloc((void **)&p_A, p_tSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory on p_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on GPU to copy the data from the host
    Cord *c_A;
    err = cudaMalloc((void **)&c_A, c_tSize * pSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory on c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(p_A, points, p_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device p_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(c_A, cords, c_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    calcCords<<<block_num, thread_num>>>(c_A, p_A);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch calcCords kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the  result from GPU to the host memory.
    err = cudaMemcpy(cords, c_A, c_tSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from c_A to cords -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(c_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(p_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

