#pragma once

#define PART  100

typedef struct Point {
    int id;
    double x1, x2;
    double a, b;
} Point;

typedef struct Info {
    int N, K, tCount;
    double D;
} Info;

Point* readPointArrayFromFile(char* fileName);
void test(int *data, int n);
int computeOnGPU(int *data, int n);
