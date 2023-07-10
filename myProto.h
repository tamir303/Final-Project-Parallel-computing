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

Point* readPointArrayFromFile(char* fileName, Info** info);
Cord* initCordsArray(Info* info, Point* points);
void test(int *data, int n);
int calcCoordinates(Cord* cords, int pSize, int cSize);
int* calcProximityCriteria(Cord* cords, double distance, int pSize, int k);
