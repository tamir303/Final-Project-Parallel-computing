#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Point *readPointArrayFromFile(char *fileName, Info **info)
{
    FILE *file = fopen(fileName, "r");
    *info = (Info *) malloc(sizeof(Info));

    if (!info)
    {
        printf("Faild to initiate info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!file)
    {
        printf("Faild to open file cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!fscanf(file, "%d %d %lf %d", &((*info)->N), &((*info)->K), &((*info)->D), &((*info)->tCount)))
    {
        printf("Faild to read info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    Point *point = (Point *)malloc(sizeof(Point) * (*info)->N);
    int count = 0;
    while (count < (*info)->N)
    {
        if (!fscanf(file, "%d %lf %lf %lf %lf", &(point[count].id), &(point[count].x1), &(point[count].x2), &(point[count].a), &(point[count].b)))
        {
            printf("Failed to read point {%d} cFunctions.c line {%d}\n", count, __LINE__);
            exit(-1);
        }
        count++;
    }

    printf("\nN: %d\nK: %d\nD: %.3f\nTcount: %d\n", (*info)->N, (*info)->K, (*info)->D, (*info)->tCount);

    return point;
}

void printResults(double *results, int counter) {
    if (counter == 0)
        printf("There were no 3 points found for any t\n");
    else
    {
        for (int i = 0; i < counter; i++)
        {
            int p1 = i * 4 + 1, p2 = i * 4 + 2, p3 = i * 4 + 3, t = i * 4;
            printf("Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", (int)results[p1],
                    (int)results[p2], (int)results[p3], results[t]);
        }
    }
}

void* allocateArray(size_t numElements, size_t elementSize) {
    void* array = malloc(numElements * elementSize);
    if (array == NULL) {
        fprintf(stderr, "Failed to allocate memory for array\n");
        exit(EXIT_FAILURE);
    }
    
    return array;
}
