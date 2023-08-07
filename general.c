#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Reads a point array from a file and populates an Info struct.
 *
 * This function reads point data from a file specified by fileName and allocates memory
 * for a Point array. It also initializes an Info struct with data read from the file.
 *
 * @param fileName The name of the file containing point data.
 * @param info A pointer to an Info pointer, which will be allocated and populated with data.
 *
 * @return A dynamically allocated Point array containing the point data.
 */
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

/**
 * @brief Prints the results to an output file.
 *
 * This function prints the results to an output file named "output.txt". It takes an array of
 * doubles (results) and the counter indicating the number of results. If the counter is zero,
 * it means no points were found satisfying the Proximity Criteria for any t.
 *
 * @param results An array of doubles containing the results.
 * @param counter The number of points that satisfy the Proximity Criteria.
 */
void printResults(double *results, int counter) {
    FILE* file;
    char filename[] = "output.txt";

    file = fopen(filename ,"w");

    if (file == NULL) {
        fprintf(stderr, "Failed to open output file\n");
        exit(EXIT_FAILURE);
    }

    if (counter == 0)
        fprintf(file, "There were no 3 points found for any t\n");
    else
    {
        for (int i = 0; i < counter; i++)
        {
            int p1 = i * 4 + 1, p2 = i * 4 + 2, p3 = i * 4 + 3, t = i * 4;
            fprintf(file, "Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", (int)results[p1],
                    (int)results[p2], (int)results[p3], results[t]);
        }
    }

    fclose(file);
}

/**
 * @brief Allocates memory for a generic array.
 *
 * This function dynamically allocates memory for a generic array of specified size.
 *
 * @param numElements The number of elements in the array.
 * @param elementSize The size of each element in bytes.
 *
 * @return A pointer to the allocated memory.
 */
void* allocateArray(size_t numElements, size_t elementSize) {
    void* array = malloc(numElements * elementSize);
    if (array == NULL) {
        fprintf(stderr, "Failed to allocate memory for array\n");
        exit(EXIT_FAILURE);
    }
    
    return array;
}
