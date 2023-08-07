#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include "general.h"

/**
 * @brief Calculates the x coordinate of a point based on the input data and user-defined function.
 *
 * This device function calculates the x coordinate of a point using the given 'x1', 'x2', and 't'
 * values. The calculation is based on a user-defined function that combines 'x1', 'x2', and 't'.
 *
 * @param x1 The x1 coordinate of the point.
 * @param x2 The x2 coordinate of the point.
 * @param t The time value for which the x coordinate is calculated.
 *
 * @return The calculated x coordinate of the point.
 */
__device__ double calcX(double x1, double x2, double t) {
    return ((x2 - x1) / 2) * sin(t * PI) + ((x2 + x1) / 2);
}

/**
 * @brief Calculates the y coordinate of a point based on the input data and user-defined function.
 *
 * This device function calculates the y coordinate of a point using the given 'x', 'a', and 'b'
 * values. The calculation is based on a user-defined function that combines 'x', 'a', and 'b'.
 *
 * @param x The x coordinate of the point.
 * @param a The 'a' coefficient of the point.
 * @param b The 'b' coefficient of the point.
 *
 * @return The calculated y coordinate of the point.
 */
__device__ double calcY(double x, double a, double b) {
    return a * x + b;
}

/**
 * @brief Calculates the coordinates of points in parallel based on input data and user-defined functions.
 *
 * This CUDA kernel function calculates the x and y coordinates of points in parallel. It takes the
 * 'cords' array of Cord structures, and each thread calculates the x and y coordinates of a subset
 * of points. The calculations are based on user-defined functions 'calcX' and 'calcY'.
 *
 * @param cords Pointer to the array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param pSize The number of points in the Cord array for each 't' value.
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

/**
 * @brief Calculates the coordinates of points in parallel and copies the result back to the host.
 *
 * This function prepares the data for parallel computation, launches the CUDA kernel 'calcCords'
 * to calculate the coordinates of points in parallel, and copies the results back to the host.
 *
 * @param cords Pointer to the array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param pSize The number of points in the Cord array for each 't' value.
 * @param cSize The number of 't' values (chunks) in the Cord array.
 *
 * @return 0 on success.
 */
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

/**
 * @brief Checks if two points are within a specified distance of each other.
 *
 * This device function checks if two points specified by their (x, y) coordinates are within
 * the specified distance 'd' of each other.
 *
 * @param x1 The x coordinate of the first point.
 * @param y1 The y coordinate of the first point.
 * @param x2 The x coordinate of the second point.
 * @param y2 The y coordinate of the second point.
 * @param d The distance threshold to check against.
 *
 * @return 1 if the two points are within the specified distance, 0 otherwise.
 */
__device__ int arePointsInDistance(double x1, double y1, double x2, double y2, double d) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy) < d;
}

/**
 * @brief Counts points that satisfy the Proximity Criteria for each 't' value in parallel.
 *
 * This CUDA kernel function counts the number of points for each 't' value that satisfy the Proximity
 * Criteria (k points within the specified distance). It takes the 'cords' array of Cord structures,
 * calculates the distances between points using 'arePointsInDistance' function, and stores the results
 * in the 'satisfiers' array.
 *
 * @param cords Pointer to the array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param satisfiers Pointer to the array where the results will be stored (1 if point satisfies the Proximity Criteria, 0 otherwise).
 * @param pSize The number of points in the Cord array for each 't' value.
 * @param distance The distance threshold for the Proximity Criteria.
 * @param k The number of points required to satisfy the Proximity Criteria.
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
                Cord PjCords = cords[tOffset + Pj];
                if (arePointsInDistance(PiCords.x, PiCords.y, PjCords.x, PjCords.y, distance))
                    count ++;
            }
        }

        satisfiers[Pi] = count >= k;
    }
}

/**
 * @brief Finds the first three points that satisfy the Proximity Criteria for each 't' value.
 *
 * This CUDA kernel function finds the first three points for each 't' value that satisfy the Proximity
 * Criteria (k points within the specified distance). It takes the 'satisfiers' array that contains the
 * results from 'countPointsInDistance' kernel, and for each 't', it finds the first three points that
 * satisfy the criteria and stores their indices in the 'output' array. It sets the corresponding 'results'
 * array value to 1 if at least three points are found, 0 otherwise.
 *
 * @param satisfiers Pointer to the array that contains the results of 'countPointsInDistance' kernel.
 * @param results Pointer to the array where the results (1 or 0) for each 't' value will be stored.
 * @param output Pointer to the array where the indices of the first three points for each 't' value will be stored.
 * @param pSize The number of points in the Cord array for each 't' value.
 */
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

/**
 * @brief Calculates the Proximity Criteria for each 't' value and returns the results.
 *
 * This function prepares the data for parallel computation, launches the CUDA kernels
 * 'countPointsInDistance' and 'findFirstThreeOnes' to calculate the Proximity Criteria for each
 * 't' value in parallel, and returns the results to the host. The results include the indices of
 * the first three points for each 't' value that satisfy the Proximity Criteria and an array
 * indicating whether at least three points were found for each 't'.
 *
 * @param cords Pointer to the array of Cord objects representing the coordinates (x and y) of points at different times.
 * @param tCount The number of 't' values (chunks) in the Cord array.
 * @param distance The distance threshold for the Proximity Criteria.
 * @param pSize The number of points in the Cord array for each 't' value.
 * @param k The number of points required to satisfy the Proximity Criteria.
 * @param output Pointer to the array where the indices of the first three points for each 't' value will be stored.
 *
 * @return An array indicating whether at least three points were found for each 't' value (1 if found, 0 otherwise).
 */
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
        fprintf(stderr, "Failed to allocate d_results - %s\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to copy data from device to host results - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(output, d_output, 3 * tCount * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device to host output - %s\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to copy data from device to host satisfiers - %s\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to free d_results - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return results;
}
