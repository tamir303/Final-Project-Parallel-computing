#pragma once

#include <mpi.h>
#include <cstddef> 
#include "myProto.h"

MPI_Datatype createInfoStruct();
MPI_Datatype createPointStruct();
MPI_Datatype createCordStruct();
MPI_Datatype createResultStruct();