#include "createMPIStruct.h"

MPI_Datatype createInfoStruct() {
   MPI_Datatype MPI_INFO;
   int infoLen[4] = {1, 1, 1 ,1};
   MPI_Aint infodis[4] = { offsetof(Info, N), offsetof(Info, K), offsetof(Info, tCount), offsetof(Info, D)};
   MPI_Datatype infotypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
   MPI_Type_create_struct(4, infoLen, infodis, infotypes, &MPI_INFO);
   MPI_Type_commit(&MPI_INFO);

   return MPI_INFO;
}

MPI_Datatype createPointStruct() {
   MPI_Datatype MPI_POINT;
   int pointLen[5] = {1, 1, 1 ,1, 1};
   MPI_Aint pointdis[5] = { offsetof(Point, id), offsetof(Point, x1), offsetof(Point, x2), offsetof(Point, a), offsetof(Point, b)};
   MPI_Datatype pointtypes[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(5, pointLen, pointdis, pointtypes, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   return MPI_POINT;
}

MPI_Datatype createCordStruct() {
   MPI_Datatype MPI_CORD;
   int cordLen[4] = {1, 1, 1 ,1};
   MPI_Aint corddis[4] = { offsetof(Cord, point), offsetof(Cord, x), offsetof(Cord, y), offsetof(Cord, t)};
   MPI_Datatype cordtypes[4] = {createPointStruct(), MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(4, cordLen, corddis, cordtypes, &MPI_CORD);
   MPI_Type_commit(&MPI_CORD);

   return MPI_CORD;
}