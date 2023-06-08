#include "mathCalc.h"

double calcX(double x1, double x2, int t) {
    return ((x2 - x1) / 2) * sin(t * M_PI) + ((x2 + x1) / 2);
}

double calcY(double x, double a, double b) {
    return a * x + b;
}

int arePointsInDistance(double x1, double y1, double x2, double y2, double d) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy) <= d;
}