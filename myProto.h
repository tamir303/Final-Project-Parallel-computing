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

typedef struct Cord {
    double x, y;
    double t;
} Cord;


Point* readPointArrayFromFile(char* fileName, Info** info);
Cord** initCords2dArray(Info* info);
void test(int *data, int n);
int computeOnGPU(Point* points, Cord* cords, int size);
