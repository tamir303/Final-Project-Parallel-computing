#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>

Point* readPointArrayFromFile(char* fileName, Info** info) {
    FILE* file = fopen(fileName, "r");
    *info = (Info*) malloc(sizeof(Info));

    if (!info) {
        printf("Faild to initiate info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!file) {
        printf("Faild to open file cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!fscanf(file, "%d %d %lf %d", &((*info)->N), &((*info)->K), &((*info)->D), &((*info)->tCount))) {
        printf("Faild to read info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    Point* point = (Point*) malloc(sizeof(Point) * (*info)->N);
    int count = 0;
    while (count < (*info)->N) {
        if (!fscanf(file, "%d %lf %lf %lf %lf",&(point[count].id), &(point[count].x1), &(point[count].x2), &(point[count].a), &(point[count].b))) {
            printf("Failed to read point {%d} cFunctions.c line {%d}\n", count, __LINE__);
            exit(-1);
        }
        printf("\nPoint %d x1: %lf x2: %lf a: %lf b: %lf",point[count].id, point[count].x1, point[count].x2, point[count].a, point[count].b);
        count++;
    }
    printf("\n");
    return point;
}

Cord* initCordsArray(Info* info, Point* points) {
    Cord* cords = (Cord*) malloc(sizeof(Cord) * (info->tCount + 1) * info->N);
    #pragma omp parallel for
    for (int tCount = 0; tCount <= info->tCount; tCount++) {
        double t = 2.0 * tCount / (info->tCount - 1.0) - 1.0;
        for (int point = 0; point < info->N; point++) {
            cords[point + tCount * info->N].point = points[point];
            cords[point + tCount * info->N].t = t;
        }
    }

    return cords;
}

void test(int *data, int n) {
    int i;
    for (i = 0;   i < n;   i++) {
        if (data[i] != i + 1) {
           printf("Wrong Calculations - Failure of the test at data[%d]\n", i);
           return;
    	}
    }
    printf("The test passed successfully\n"); 
}
