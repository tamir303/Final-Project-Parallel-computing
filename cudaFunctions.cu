#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include "mathCalc.h"

/**
 * Calculates the coordinates of points based on input data and user-defined functions.
 *
 * @param cords         Pointer to an array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param points        Pointer to an array of Point objects containing information about initial positions and coefficients.
 * @param fPcalcX       Function pointer to a user-defined function that calculates the x coordinate.
 *                      Signature: double functionName(double x1, double x2, double t)
 * @param fPcalcY       Function pointer to a user-defined function that calculates the y coordinate.
 *                      Signature: double functionName(double xCord, double a, double b)
 */
__global__ void calcCords(
    Cord* cords,
    Point* points,
    double (*fPcalcX)(double, double, double),
    double (*fPcalcY)(double, double, double)
) {
    // Get the indices of the current thread and the total number of threads
    int point = threadIdx.x;
    int tCount = blockIdx.x;
    int offset = blockDim.x;
    
    // Retrieve the necessary values from the input arrays
    double x1 = points[point].x1;
    double x2 = points[point].x2;
    double t = cords[tCount].t;
    double xCord;
    double a = points[point].a;
    double b = points[point].b;
    
    // Calculate the x coordinate using the user-defined function
    xCord = cords[tCount * offset + point].x = fPcalcX(x1, x2, t);
    
    // Calculate the y coordinate using the user-defined function
    cords[tCount * offset + point].y = fPcalcY(xCord, a, b);
}



int computeOnGPU(Point* points, Cord* cords, int pSize, int cSize) {
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

    printf("this is fine\n");

    // Allocate memory on GPU to copy the data from the host
    Cord *c_A;
    err = cudaMalloc((void **)&c_A, c_tSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory on c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

        printf("this is fine\n");


    // Copy data from host to the GPU memory
    err = cudaMemcpy(p_A, points, p_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device p_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("this is fine\n");

    // Copy data from host to the GPU memory
    err = cudaMemcpy(c_A, cords, c_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("this is fine\n");

    calcCords<<<block_num, thread_num>>>(c_A, p_A, calcX, calcY);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch calcCords kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("this is fine\n");

    // Copy the  result from GPU to the host memory.
    err = cudaMemcpy(cords, c_A, c_tSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from c_A to cords -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("this is fine\n");

    // Free allocated memory on GPU
    if (cudaFree(c_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("this is fine\n");

    // Free allocated memory on GPU
    if (cudaFree(p_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

