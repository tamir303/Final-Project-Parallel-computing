#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "parallelFunctions.h"
#include "general.h"

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
                printf("%d %d %d\n", tPoints[0], tPoints[1], tPoints[2]);
                memcpy(&dest[local_counter * (PCT + 1)], res, sizeof(double) * (PCT + 1));
                local_counter++;
            }
        }
    }

   return local_counter;
}


