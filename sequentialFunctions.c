#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sequentialFunctions.h"
#include "general.h"

/**
 * @brief Initializes the Cord array sequentially.
 *
 * This function initializes the Cord array sequentially. It creates a 2D array of Cord structures
 * to store point data with different values of 't'. The 't' values are calculated based on the
 * number of intervals (info->tCount) and evenly distributed between -1.0 and 1.0. Each 't' value
 * corresponds to a region of Cord structures initialized with the given 'points' data.
 *
 * @param info Pointer to the Info struct containing relevant information.
 * @param points Pointer to the array of Point structures to be used for initialization.
 *
 * @return The initialized Cord array with point data at different 't' values.
 */
Cord *initCordsArraySequential(Info *info, Point *points)
{
    Cord *cords = (Cord *) allocateArray((info->tCount + 1) * info->N, sizeof(Cord));

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

/**
 * @brief Calculates the x and y coordinates of points in the Cord array sequentially.
 *
 * This function calculates the x and y coordinates of points in the Cord array sequentially.
 * It iterates through each 't' value and calculates the corresponding x and y coordinates based
 * on the equations specified in the Point structure. The results are stored in the same Cord array.
 *
 * @param cords Pointer to the Cord array containing point data at different 't' values.
 * @param info Pointer to the Info struct containing relevant information.
 */
void calcCordsSequential(Cord *cords, Info *info)
{
    for (int tCount = 0; tCount <= info->tCount; tCount++)
    {
        int offset = tCount * info->N;
        for (int p = 0; p < info->N; p++)
        {
            double x1 = cords[offset + p].point.x1, x2 = cords[offset + p].point.x2, t = cords[offset + p].t;
            double a = cords[offset + p].point.a, b = cords[offset + p].point.b;
            cords[offset + p].x = ((x2 - x1) / 2) * sin(t * PI) + ((x2 + x1) / 2);
            cords[offset + p].y = a * cords[offset + p].x + b;
        }
    }
}

/**
 * @brief Finds points satisfying Proximity Criteria sequentially.
 *
 * This function finds points that satisfy the Proximity Criteria sequentially. It iterates through
 * each 't' value and checks for groups of three points that satisfy the criteria. If found, the
 * information of the three points is stored in the 'dest' array for further analysis.
 *
 * @param src Pointer to the Cord array containing point data at different 't' values.
 * @param dest Pointer to the array where the results will be stored.
 * @param info Pointer to the Info struct containing relevant information.
 *
 * @return The number of groups of three points that satisfy the Proximity Criteria.
 */
int findProximityCriteriaSequential(Cord *src, double *dest, Info *info)
{
    int count_satisfy = 0;
    for (int tCount = 0; tCount <= info->tCount; tCount++)
    {
        double threePoints[4] = {0};
        int offset = tCount * info->N, count_in_distance = 0, count_group = 1;
        Cord *tCountRegion = src + offset;
        threePoints[0] = tCountRegion[0].t;

        for (int Pi = 0; Pi < info->N && count_group < 4; Pi++)
        {
            for (int Pj = 0; Pj < info->N && count_in_distance < info->K; Pj++)
            {
                if (Pi != Pj)
                {
                    double dx = tCountRegion[Pj].x - tCountRegion[Pi].x;
                    double dy = tCountRegion[Pj].y - tCountRegion[Pi].y;
                    if (sqrt(dx * dx + dy * dy) < info->D)
                        count_in_distance++;
                }
            }

            if (count_in_distance >= info->K)
            {
                threePoints[count_group] = (double)tCountRegion[Pi].point.id;
                count_group++;
            }

            count_in_distance = 0;
        }

        if (count_group == 4)
        {
            memcpy(&dest[count_satisfy * 4], threePoints, sizeof(threePoints));
            count_satisfy++;
        }

        count_group = 1;
    }

    return count_satisfy;
}
