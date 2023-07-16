#pragma once

#include "myProto.h"

int findProximityCriteria(Cord* src, double *dest, Info* info, int chunkSize);
Cord *initCordsArray(Info *info, Point *points);