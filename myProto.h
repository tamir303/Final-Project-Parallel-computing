#pragma once

#define PART  100

typedef struct Point {
    int id;
    double x1, x2;
    double a, b;
} Point;

void test(int *data, int n);
int computeOnGPU(int *data, int n);
