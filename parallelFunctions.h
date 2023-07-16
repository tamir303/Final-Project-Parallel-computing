#pragma once

#include "myProto.h"

int findProximityCriteriaParallel(Cord* src, double *dest, Info* info, int chunkSize);
Cord *initCordsArrayParallel(Info *info, Point *points);