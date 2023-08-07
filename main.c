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

/**
 * @brief Main function of the MPI+OpenMP+CUDA integration example.
 * 
 * The main function reads command-line arguments to determine whether to run the parallel or sequential implementation.
 * It initializes MPI, gets the size and rank of the current process, and performs the appropriate implementation.
 * The parallel implementation uses MPI to distribute data among processes, OpenMP to perform local calculations, and CUDA for GPU acceleration.
 * The sequential implementation performs all calculations sequentially without parallelization.
 * The execution time and results are printed after each implementation.
 * 
 * @param argc The number of command-line arguments.
 * @param argv An array of strings containing the command-line arguments.
 * @return 0 on successful execution, non-zero otherwise.
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


/**
 * @brief Run the parallel execution of the program.
 *
 * The parallel function initializes MPI, exchanges information between processes,
 * reads data from the input file, calculates the coordinates of points for each time step
 * in parallel using OpenMP and CUDA, finds points that satisfy the Proximity Criteria,
 * and gathers the results to the master process for printing.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return void
 */
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


/**
 * @brief Run the sequential execution of the program.
 *
 * The sequential function reads data from the input file, prepares the Cord array,
 * calculates the coordinates of points for each time step sequentially,
 * finds points that satisfy the Proximity Criteria, and prints the results.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return void
 */
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
