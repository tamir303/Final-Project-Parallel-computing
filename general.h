#pragma once

#include "myProto.h"

void printResults(double *results, int counter);
Point* readPointArrayFromFile(char* fileName, Info** info);
void* allocateArray(size_t numElements, size_t elementSize);