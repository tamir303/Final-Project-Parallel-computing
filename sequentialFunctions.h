#pragma once

#include "myProto.h"

int findProximityCriteriaSequential(Cord* src, double *dest, Info* info);
Cord *initCordsArraySequential(Info *info, Point *points);
void calcCordsSequential(Cord* cords, Info* info);
