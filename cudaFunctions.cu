#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include "general.h"

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
 */
__global__ void calcCords(Cord* cords, int pSize) {
    // Get the indices of the current thread and the total number of threads
    int tCount = blockIdx.x;
    int offset = pSize * tCount;
    int start = threadIdx.x * (pSize / blockDim.x);
    int end = blockDim.x - 1 == threadIdx.x ? pSize : start + pSize / blockDim.x;

    Point p;
    double t, xCord;
    for (int i = start; i < end; i++) {
        p = cords[i + offset].point;
        t = cords[i + offset].t;

        // Calculate the x coordinate using the user-defined function
        xCord = cords[i + offset].x = calcX(p.x1, p.x2, t);

        // Calculate the y coordinate using the user-defined function
        cords[i + offset].y = calcY(xCord, p.a, p.b);
    }
}

int calcCoordinates(Cord* cords, int pSize, int cSize) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t c_tSize = cSize * pSize * sizeof(Cord);
    int thread_num = 100;
    int block_num = cSize;

    // Allocate memory on GPU to copy the data from the host
    Cord *c_A;
    err = cudaMalloc((void **)&c_A, c_tSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory on c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(c_A, cords, c_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    calcCords<<<block_num, thread_num>>>(c_A, pSize);
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

    return 0;
}

// ################################################################################### //


__device__ int arePointsInDistance(double x1, double y1, double x2, double y2, double d) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy) < d;
}

/**
 * Calculates the coordinates of points based on input data and user-defined functions.
 *
 * @param cords         Pointer to an array of Cord objects representing the coordinates (x and y) of points at different times.
 */
__global__ void countPointsInDistance(Cord* cords, int* satisfiers, int pSize, double distance, int k) {
    // Get the indices of the current thread and the total number of threads
    int Pi, Pj, count;
    int tOffset = pSize * blockIdx.x;
    int start = tOffset + threadIdx.x * (pSize / blockDim.x);
    int end = blockDim.x - 1 == threadIdx.x ? tOffset + pSize : start + pSize / blockDim.x;

    for (Pi = start; Pi < end; Pi++) {
        count = 0;
        for (Pj = 0; Pj < pSize && count < k ; Pj++) {
            if (Pi != Pj) {
                Cord PiCords = cords[Pi];
                Cord PjCords = cords[Pj];
                if (arePointsInDistance(PiCords.x, PiCords.y, PjCords.x, PjCords.y, distance))
                    count ++;
            }
        }

        satisfiers[Pi] = count >= k;
    }
}

__global__ void findFirstThreeOnes(const int* satisfiers, int* results, int* output, int pSize) {

    __shared__ int counter;
    if (threadIdx.x == 0)
        counter = 0;

    __syncthreads();

    int tOffset = pSize * blockIdx.x;
    int start = tOffset + threadIdx.x * (pSize / blockDim.x);
    int end = blockDim.x - 1 == threadIdx.x ? tOffset + pSize : start + pSize / blockDim.x;

    for (int i = start; i < end; i++) {
        if (satisfiers[i] == 1) {
            int currIndex = atomicAdd(&counter, 1);
            if (currIndex < 3) {
                atomicExch(&output[blockIdx.x * 3 + currIndex], i % pSize);
            }
            else if (currIndex >= 3) {
                atomicExch(&results[blockIdx.x], 1);
                break;
            }
        }
    }
}

int* calcProximityCriteria(Cord* cords, int tCount, double distance, int pSize, int k, int* output) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t c_tSize =  pSize * tCount * sizeof(Cord);
    size_t s_tSize = pSize * tCount * sizeof(int);
    int thread_num = 100;
    int block_num = tCount;

    // Allocate memory on GPU to copy the data from the host
    int *s_A;
    err = cudaMalloc((void **)&s_A, s_tSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory on s_A - %s\n", cudaGetErrorString(err));
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
    err = cudaMemcpy(c_A, cords, c_tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    countPointsInDistance<<<block_num, thread_num>>>(c_A, s_A, pSize, distance, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch countPointsInDistance kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    int* d_output;
    int* d_results, *results = NULL;

    err = cudaMalloc((void**)&d_output, 3 * tCount * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_output - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_results, tCount * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_output - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    findFirstThreeOnes<<<block_num, thread_num>>>(s_A, d_results, d_output, pSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch findFirstThreeOnes kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Copy data from host to the GPU memory
    results = (int*) allocateArray(tCount, sizeof(int));
    err = cudaMemcpy(results, d_results, tCount * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(output, d_output, 3 * tCount * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(c_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free c_A- %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int* satisfiers = (int*)allocateArray(tCount * pSize, sizeof(int));
    // Copy data from host to the GPU memory
    err = cudaMemcpy(satisfiers, s_A, s_tSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device c_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(s_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free s_A - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(d_output) != cudaSuccess) {
        fprintf(stderr, "Failed to free d_output - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(d_results) != cudaSuccess) {
        fprintf(stderr, "Failed to free d_output - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return results;
}
