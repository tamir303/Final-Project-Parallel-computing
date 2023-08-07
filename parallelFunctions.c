#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "parallelFunctions.h"
#include "general.h"

/**
 * @brief Initializes the Cord array in parallel.
 *
 * This function initializes the Cord array in parallel. It creates a 2D array of Cord structures
 * to store point data with different values of 't'. The 't' values are calculated based on the
 * number of threads (info->tCount) and evenly distributed between -1.0 and 1.0. Each thread takes
 * responsibility for a subset of 't' values and initializes the corresponding Cord structures
 * with the given 'points' data.
 *
 * @param info Pointer to the Info struct containing relevant information.
 * @param points Pointer to the array of Point structures to be used for initialization.
 *
 * @return The initialized Cord array with point data at different 't' values.
 */
Cord *initCordsArrayParallel(Info *info, Point *points)
{
    Cord *cords = (Cord *) allocateArray((info->tCount + 1) * info->N, sizeof(Cord));

    #pragma omp parallel for num_threads(info->tCount)
    for (int tCount = 0; tCount <= info->tCount; tCount++)
    {
        double t = 2.0 * tCount / info->tCount - 1.0;

        #pragma omp parallel for num_threads(info->N)
        for (int point = 0; point < info->N; point++)
        {
            cords[point + tCount * info->N].point = points[point];
            cords[point + tCount * info->N].t = t;
        }
    }

    return cords;
}

/**
 * @brief Finds points satisfying Proximity Criteria in parallel.
 *
 * This function finds points that satisfy the Proximity Criteria in parallel. It takes the
 * source Cord array 'src' with data at different 't' values, 'info' containing relevant information,
 * and the size of the 'chunkSize' to be processed concurrently. The function calculates Proximity
 * Criteria for each 't' value in parallel and stores the results in the 'dest' array. The 'dest'
 * array is a one-dimensional array of doubles that holds the results for each 't'. If a set of
 * points satisfies the Proximity Criteria at a specific 't', their corresponding information is
 * stored in the 'dest' array for further analysis.
 *
 * @param src Pointer to the Cord array containing data at different 't' values.
 * @param dest Pointer to the array where the results will be stored.
 * @param info Pointer to the Info struct containing relevant information.
 * @param chunkSize The size of the chunks to be processed concurrently.
 *
 * @return The number of 't' values for which points satisfy the Proximity Criteria.
 */
int findProximityCriteriaParallel(Cord *src, double *dest, Info* info, int chunkSize)
{
    int local_counter = 0;
    int *satisfiers = (int*) allocateArray(chunkSize * 3, sizeof(int));
    int *results = calcProximityCriteria(src, chunkSize, info->D, info->N, info->K, satisfiers);

    #pragma omp parallel for num_threads(chunkSize) schedule(dynamic)
    for (int tCount = 0; tCount < chunkSize; tCount++)
    {
        int* tPoints = satisfiers + tCount * 3;
        if (results[tCount] == 1) {
            #pragma omp critical
            {
                double t = src[info->N * tCount].t;
                double* res = (double*) allocateArray(PCT + 1, sizeof(double));
                res[0] = t; res[1] = (double) tPoints[0]; res[2] = (double) tPoints[1]; res[3] = (double) tPoints[2];
                memcpy(&dest[local_counter * (PCT + 1)], res, sizeof(double) * (PCT + 1));
                local_counter++;
            }
        }
    }

   return local_counter;
}


