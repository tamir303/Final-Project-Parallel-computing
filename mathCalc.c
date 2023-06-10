#include "mathCalc.h"

int arePointsInDistance(double x1, double y1, double x2, double y2, double d) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy) <= d;
}