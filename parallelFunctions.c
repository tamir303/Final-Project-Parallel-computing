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
    #pragma omp parallel for
    for (int tCount = 0; tCount <= info->tCount; tCount++)
    {
        double t = 2.0 * tCount / (info->tCount) - 1.0;
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
    #pragma omp parallel num_threads(chunkSize)
    {
        // Iterate over data chunk assigned to each thread
        int count_group = 1;
        int offset = omp_get_thread_num() * info->N;
        double threePoints[PCT + 1] = {0};
        Cord *tCountRegion = src + offset;
        threePoints[0] = tCountRegion[0].t;

        // Iterate over each region of points per t
        int *satisfiers = calcProximityCriteria(tCountRegion, info->D, info->N, info->K);

        #pragma omp parallel for
        for (int i = 0; i < info->N; i++)
            if (satisfiers[i])
                #pragma omp critical
                {
                if (count_group < PCT + 1)
                {
                    threePoints[count_group] = (double)tCountRegion[i].point.id;
                    count_group++;
                }
                }

        if (count_group == PCT + 1)
        {
            #pragma omp critical
            {
                memcpy(&dest[local_counter * (PCT + 1)], threePoints, sizeof(threePoints));
                local_counter++;
            }
        }
    }

   return local_counter;
}


