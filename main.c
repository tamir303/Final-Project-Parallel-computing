#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "myProto.h"
#include "parallelFunctions.h"
#include "sequentialFunctions.h"
#include "createMPIStruct.h"
#include "general.h"

/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/

int main(int argc, char *argv[])
{
   if (strcmp(argv[2], "par") == 0)
      parallel(argc, argv);
   else if (strcmp(argv[2], "seq") == 0)
      sequential(argc, argv);
   else
      printf("\nUndefined Case, seconds argument has to be 'seq' or 'par'\n");

   return 0;
}

// ########################################################## PARALLEL ##################################################

void parallel(int argc, char *argv[])
{
   int size, rank;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Info *info = NULL;
   Point *points = NULL;
   Cord *data = NULL, *cords = NULL;
   MPI_Datatype MPI_INFO = createInfoStruct(), MPI_CORD = createCordStruct();

   // Read data and exchange information between processes

   clock_t start, end;
   start = clock();
   double exec_time;

   if (rank == 0)
   {
      points = readPointArrayFromFile(argv[1], &info);
      data = initCordsArrayParallel(info, points);
      MPI_Send(info, 1, MPI_INFO, 1, 0, MPI_COMM_WORLD);
   }
   else
   {
      info = (Info *) allocateArray(1, sizeof(Info));
      MPI_Recv(info, 1, MPI_INFO, 0, 0, MPI_COMM_WORLD, &status);
   }

   int chunkSize = rank == size - 1 ? (int) ceil(info->tCount / size) : info->tCount / size;
   cords = (Cord *) allocateArray(chunkSize * info->N, sizeof(Cord));
   MPI_Scatter(data, chunkSize * info->N, MPI_CORD, cords, chunkSize * info->N, MPI_CORD, 0, MPI_COMM_WORLD);

   if (calcCoordinates(cords, info->N, chunkSize) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   // Assuming cords is populated with data
   // Divide the data into chunks

   // OpenMP parallel region
   double *local_results = (double *) allocateArray(chunkSize * (PCT + 1), sizeof(double)), *global_results = NULL;
   int local_counter, global_counter = 0;

   local_counter = findProximityCriteriaParallel(cords, local_results, info, chunkSize);

   MPI_Barrier(MPI_COMM_WORLD);

   if (rank == 0)
      global_results = (double *) allocateArray(info->tCount * (PCT + 1), sizeof(double));

   // Communicate results to the master process
   MPI_Gather(local_results, local_counter * (PCT + 1), MPI_DOUBLE, global_results, local_counter * (PCT + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Reduce(&local_counter, &global_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   end = clock();
   exec_time = (double)(end - start) / CLOCKS_PER_SEC;

   if (rank == 0)
   {
      printResults(global_results, global_counter);
      printf("Execution time: %.6f seconds\n", exec_time);
   }

   MPI_Type_free(&MPI_INFO);
   MPI_Type_free(&MPI_CORD);
   MPI_Finalize();
}

// ########################################################## SEQUENTIAL ##################################################

void sequential(int argc, char *argv[])
{
   Info *info = NULL;
   Point *points = NULL;
   Cord *cords = NULL;

   points = readPointArrayFromFile(argv[1], &info);

   clock_t start, end;
   start = clock();
   double exec_time;

   // Prepare Cords Array
   cords = initCordsArraySequential(info, points);

   // Calculate each Point Cord's per t
   calcCordsSequential(cords, info);

   // Find Points that satisfy Proximity Criteria
   int count_satisfy;
   double *results = (double *) allocateArray(4 * (info->tCount + 1), sizeof(double));

   count_satisfy = findProximityCriteriaSequential(cords, results, info);

   end = clock();
   exec_time = (double)(end - start) / CLOCKS_PER_SEC;

   printResults(results, count_satisfy);

   printf("Execution time: %.6f seconds\n", exec_time);
}
