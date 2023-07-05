#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"
#include "mathCalc.h"
#include <cstddef>  // Add this line to include the <cstddef> header

/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/

int main(int argc, char *argv[]) {
   int size, rank;
   MPI_Status  status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Info* info = NULL;
   Point* points = NULL;
   Cord* data = NULL;
   Cord* cords = NULL;
   MPI_Datatype MPI_POINT;
   MPI_Datatype MPI_INFO;
   MPI_Datatype MPI_CORD;

   // Create MPI Struct for Point
   int pointLen[5] = {1, 1, 1 ,1, 1};
   MPI_Aint pointdis[5] = { offsetof(Point, id), offsetof(Point, x1), offsetof(Point, x2), offsetof(Point, a), offsetof(Point, b)};
   MPI_Datatype pointtypes[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(5, pointLen, pointdis, pointtypes, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);
   
   // Create MPI Struct for Info
   int infoLen[4] = {1, 1, 1 ,1};
   MPI_Aint infodis[4] = { offsetof(Info, N), offsetof(Info, K), offsetof(Info, tCount), offsetof(Info, D)};
   MPI_Datatype infotypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
   MPI_Type_create_struct(4, infoLen, infodis, infotypes, &MPI_INFO);
   MPI_Type_commit(&MPI_INFO);

   // Create MPI Struct for Cord
   int cordLen[5] = {1, 1, 1, 1 ,1};
   MPI_Aint corddis[5] = { offsetof(Point, point), offsetof(Cord, id), offsetof(Cord, x), offsetof(Cord, y), offsetof(Cord, t)};
   MPI_Datatype cordtypes[5] = {MPI_POINT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
   MPI_Type_create_struct(5, cordLen, corddis, cordtypes, &MPI_CORD);
   MPI_Type_commit(&MPI_CORD);

   // Read data and exchange information between processes
   if (rank == 0) {
      points = readPointArrayFromFile(argv[1], &info);
      data = initCordsArray(info);
      MPI_Send(info, 1, MPI_INFO, 1, 0, MPI_COMM_WORLD);
   } else {
      info = (Info*) malloc(sizeof(Info));
      MPI_Recv(info, 1, MPI_INFO, 0, 0, MPI_COMM_WORLD, &status);
   }

   int chunkSize = rank == size - 1 ? (int) ceil(info->tCount / size) : info->tCount / size;
   cords = (Cord*) malloc(sizeof(Cords) * chunkSize);
   MPI_Scatter(data, chunkSize, MPI_CORD, cords, chunkSize, MPI_CORD, 0, MPI_COMM_WORLD);

   // Perform computations of Cords with CUDA and OpenMP
   /**
    * data - Array of Points size info->N / processes
    * cords - 2d Array of Cords per t(i)
    * return -> Calculate all Cords for t(i) = 2 * i / tCount - 1
   */
   if (computeOnGPU(cords, info->N, chunkSize) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   MPI_Barrier(MPI_COMM_WORLD);

   for (int i = 0; i < info->tCount * (info->N / 2); i++)
      printf("proc: %d, pointID: %d, t: %lf, x: %lf, y: %lf\n", rank, cords[i].id, cords[i].t, cords[i].x, cords[i].y);

   int chunk_size = info->tCount * info->N / size; // Number of elements in each process
   int found_count = 0;
   int found_flag = 0;
   int result_flag = 0;

   // Assuming cords is populated with data
   // Divide the data into equal-sized chunks
   MPI_Gather(cords, chunk_size, MPI_CORD, allCords, chunk_size, MPI_CORD, 0, MPI_COMM_WORLD);

   // OpenMP parallel region
   #pragma omp parallel num_threads(info->tCount)
   {
      // Iterate over data chunk assigned to each thread
      for (int i = 0; i < chunk_size; i++) {
         // Iterate over K cords for each t
         for (int j = 0; j < info->K; j++) {
               // Check distance condition
               if (arePointsInDistance(cords[i].x, cords[i].y, cords[j].x, cords[j].y, info->D)) {
                  // Set flag indicating a point has been found
                  found_flag = 1;
               }
         }
      }
   }

   // Communicate results to the master process
   if (rank != 0) {
      // Send the flag to the master process with tag 0
      MPI_Send(&found_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
   } else {
      // Receive flags from other processes
      for (int i = 1; i < size; i++) {
         // Receive the flag with tag i
         MPI_Recv(&result_flag, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         if (result_flag == 1) {
               found_count++;
               if (found_count >= 3) {
                  // Send termination signal to all processes
                  for (int j = 0; j < size; j++) {
                     MPI_Send(&found_flag, 1, MPI_INT, j, size + 1, MPI_COMM_WORLD);
                  }
                  break;
               }
         }
      }
   }

   // Terminate if termination signal received
   if (found_count >= 3) {
      MPI_Finalize();
      return 0;
   }


   MPI_Type_free(&MPI_POINT);
   MPI_Type_free(&MPI_INFO);
   MPI_Finalize();

   return 0;
}


