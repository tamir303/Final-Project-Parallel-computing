#pragma once

#define PI 3.14159
#define PCT 3 // Number of points to satisfy the proximity criteria

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
    Point point;
    double x, y;
    double t;
} Cord;

// Run Options
void sequential(int argc, char *argv[]);
void parallel(int argc, char *argv[]);

// CUDA Functions
int calcCoordinates(Cord* cords, int pSize, int cSize);
int* calcProximityCriteria(Cord* cords, int tCount, double distance, int pSize, int k, int* output);
