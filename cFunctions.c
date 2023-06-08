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

Cord** initCords2dArray(Info* info) {
    Cord** cords = (Cord**) malloc(sizeof(Cord*) * info->N);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < info->N; i++) {
        cords[i] = (Cord*) malloc(sizeof(Cord) * (info->tCount + 1));
        for (int j = 0; j <= info->tCount; j++) {
            cords[i][j].t = 2.0 * i / info->tCount - 1.0;
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
