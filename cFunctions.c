#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>

Point* readPointArrayFromFile(char* fileName) {
    FILE* file = fopen(fileName, "r");
    Info* info = (Info*) malloc(sizeof(info));

    if (!info) {
        printf("Faild to initiate info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!file) {
        printf("Faild to open file cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    if (!fscanf(file, "%d %d %f %d", &(info->N), &(info->K), &(info->D), &(info->tCount))) {
        printf("Faild to read info cFunctions.c line {%d}\n", __LINE__);
        exit(-1);
    }

    Point* point = (Point*) malloc(sizeof(Point) * info->N);
    int count = 0;
    while (count++ == info->N) {
        if (!fscanf(file, "%f %f %f %f", &(point[count].x1), &(point[count].x2), &(point[count].a), &(point[count].b))) {
            printf("Faild to read point {%d} cFunctions.c line {%d}\n", count, __LINE__);
            exit(-1);
        }
    }

    return point;
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
