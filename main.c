#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"
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

   Info* info = NULL;
   Point* data = NULL;
   MPI_Datatype MPI_POINT;
   MPI_Datatype MPI_INFO;

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

   if (size != 2) {
      printf("Run the example with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // Read data and exchange information between processes
   if (rank == 0) {
      data = readPointArrayFromFile(argv[1], &info);
      MPI_Send(info, 1, MPI_INFO, 1, 0, MPI_COMM_WORLD);
      MPI_Send(data + (info->N / size), info->N / size, MPI_POINT, 1, 0, MPI_COMM_WORLD);
   } else {
      info = (Info*) malloc(sizeof(Info));
      MPI_Recv(info, 1, MPI_INFO, 0, 0, MPI_COMM_WORLD, &status);
      data = (Point *) malloc(sizeof(Point) * (info->N / size));
      MPI_Recv(data, (info->N / size), MPI_POINT, 0, 0, MPI_COMM_WORLD, &status);
   }

   // Initialize 2d array of cords
   Cord* cords = initCordsArray(info);

   // Perform computations of Cords with CUDA and OpenMP
   /**
    * data - Array of Points size info->N / processes
    * cords - 2d Array of Cords per t(i)
    * return -> Calculate all Cords for t(i) = 2 * i / tCount - 1
   */
   if (computeOnGPU(data, cords, info->N / size, info->tCount) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);


   int chunk_size; // Number of elements in each process
   int found_count = 0;
   int found_flag = 0;
   int result_flag = 0;


    // Distribute data among processes
    if (rank == 0) {
        // Assuming cords is populated with data
        // Divide the data into equal-sized chunks
        chunk_size = (info->tCount * info->N) / size;
        // Scatter the chunks to each process
        MPI_Scatter(cords, chunk_size, MPI_DOUBLE, cords, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // Receive the scattered chunks
        MPI_Scatter(cords, chunk_size, MPI_DOUBLE, cords, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

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


