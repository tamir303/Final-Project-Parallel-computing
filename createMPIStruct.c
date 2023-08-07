#include "createMPIStruct.h"

/**
 * @brief Creates a custom MPI datatype for the Info struct.
 *
 * This function creates a custom MPI datatype that corresponds to the structure Info.
 * The Info struct contains four elements: N, K, tCount, and D. This function creates
 * an MPI datatype representing the Info struct so that it can be efficiently communicated
 * among MPI processes.
 *
 * @return The custom MPI datatype for the Info struct.
 */
MPI_Datatype createInfoStruct() {
   MPI_Datatype MPI_INFO;
   int infoLen[4] = {1, 1, 1 ,1};
   MPI_Aint infodis[4] = { offsetof(Info, N), offsetof(Info, K), offsetof(Info, tCount), offsetof(Info, D)};
   MPI_Datatype infotypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
   MPI_Type_create_struct(4, infoLen, infodis, infotypes, &MPI_INFO);
   MPI_Type_commit(&MPI_INFO);

   return MPI_INFO;
}

/**
 * @brief Creates a custom MPI datatype for the Point struct.
 *
 * This function creates a custom MPI datatype that corresponds to the structure Point.
 * The Point struct contains five elements: id, x1, x2, a, and b. This function creates
 * an MPI datatype representing the Point struct so that it can be efficiently communicated
 * among MPI processes.
 *
 * @return The custom MPI datatype for the Point struct.
 */
MPI_Datatype createPointStruct() {
   MPI_Datatype MPI_POINT;
   int pointLen[5] = {1, 1, 1 ,1, 1};
   MPI_Aint pointdis[5] = { offsetof(Point, id), offsetof(Point, x1), offsetof(Point, x2), offsetof(Point, a), offsetof(Point, b)};
   MPI_Datatype pointtypes[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(5, pointLen, pointdis, pointtypes, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   return MPI_POINT;
}

/**
 * @brief Creates a custom MPI datatype for the Cord struct.
 *
 * This function creates a custom MPI datatype that corresponds to the structure Cord.
 * The Cord struct contains four elements: point, x, y, and t. The 'point' element itself
 * is of the custom Point struct type. This function creates an MPI datatype representing
 * the Cord struct so that it can be efficiently communicated among MPI processes.
 *
 * @return The custom MPI datatype for the Cord struct.
 */
MPI_Datatype createCordStruct() {
   MPI_Datatype MPI_CORD;
   int cordLen[4] = {1, 1, 1 ,1};
   MPI_Aint corddis[4] = { offsetof(Cord, point), offsetof(Cord, x), offsetof(Cord, y), offsetof(Cord, t)};
   MPI_Datatype cordtypes[4] = {createPointStruct(), MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(4, cordLen, corddis, cordtypes, &MPI_CORD);
   MPI_Type_commit(&MPI_CORD);

   return MPI_CORD;
}