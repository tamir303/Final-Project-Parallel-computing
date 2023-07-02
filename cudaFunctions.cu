#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ double calcX(double x1, double x2, double t) {
    return ((x2 - x1) / 2) * sin(t * PI) + ((x2 + x1) / 2);
}

__device__ double calcY(double x, double a, double b) {
    return a * x + b;
}

/**
 * Calculates the coordinates of points based on input data and user-defined functions.
 *
 * @param cords         Pointer to an array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param points        Pointer to an array of Point objects containing information about initial positions and coefficients.S
 */
__global__ void calcCords(Cord* cords, Point* points, int pointOffset) {
    // Get the indices of the current thread and the total number of threads
    int point = pointOffset + threadIdx.x;
    int tCount = blockIdx.x;
    int offset = blockDim.x;
    
    // Retrieve the necessary values from the input arrays
    double x1 = points[point].x1;
    double x2 = points[point].x2;
    double t = cords[tCount].t;
    double xCord;
    double a = points[point].a;
    double b = points[point].b;

    // Set Cord PointID
    cords[tCount * offset + point].id = point;
    
    // Calculate the x coordinate using the user-defined function
    xCord = cords[tCount * offset + point].x = calcX(x1, x2, t);
    
    // Calculate the y coordinate using the user-defined function
    cords[tCount * offset + point].y = calcY(xCord, a, b);
}



int computeOnGPU(Point* points, Cord* cords, int pSize, int cSize, int pointOffset) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t p_tSize = pSize * sizeof(Point);
    size_t c_tSize = cSize * pSize * sizeof(Cord);
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
    err = cudaMalloc((void **)&c_A, c_tSize);
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

    calcCords<<<block_num, thread_num>>>(c_A, p_A, pointOffset);
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

