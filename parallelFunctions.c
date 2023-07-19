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
    #pragma omp parallel for num_threads(chunkSize) reduction(+:local_counter)
    for (int i = 0; i < chunkSize; i++)
    {
        // Iterate over data chunk assigned to each thread
        int offset = omp_get_thread_num() * info->N;
        Cord *tCountRegion = src + offset;

        // Iterate over each region of points per t
        int *satisfiers = calcProximityCriteria(tCountRegion, info->D, info->N, info->K);
        if (satisfiers != NULL) {
            satisfiers[PCT] = tCountRegion[0].t;
            printf("%d %d %d\n", satisfiers[0], satisfiers[1], satisfiers[2]);
            memcpy(&dest[omp_get_thread_num() * (PCT + 1)], satisfiers, sizeof(int) * (PCT + 1));
            local_counter++;
        }
    }

   return local_counter;
}


