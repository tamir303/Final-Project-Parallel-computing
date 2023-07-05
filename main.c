#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"
#include "mathCalc.h"
#include "createMPIStruct.h"
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
   MPI_Datatype MPI_INFO = createInfoStruct();
   MPI_Datatype MPI_CORD = createCordStruct();

   // Read data and exchange information between processes
   if (rank == 0) {
      points = readPointArrayFromFile(argv[1], &info);
      data = initCordsArray(info, points);
      MPI_Send(info, 1, MPI_INFO, 1, 0, MPI_COMM_WORLD);
   } else {
      info = (Info*) malloc(sizeof(Info));
      MPI_Recv(info, 1, MPI_INFO, 0, 0, MPI_COMM_WORLD, &status);
   }

   int chunkSize = rank == size - 1 ? (int) ceil(info->tCount / size) : info->tCount / size;
   cords = (Cord*) malloc(sizeof(Cord) * chunkSize * info->N);
   MPI_Scatter(data, chunkSize * info->N , MPI_CORD, cords, chunkSize * info->N, MPI_CORD, 0, MPI_COMM_WORLD);

   if (computeOnGPU(cords, info->N, chunkSize) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   MPI_Barrier(MPI_COMM_WORLD);

   for (int i = 0; i < info->tCount / size * info->N; i++)
      printf("proc: %d, pointID: %d, t: %lf, x: %lf, y: %lf\n", rank, cords[i].point.id, cords[i].t, cords[i].x, cords[i].y);

   int found_count = 0;
   int found_flag = 0;
   int result_flag = 0;

   // Assuming cords is populated with data
   // Divide the data into equal-sized chunks

   // OpenMP parallel region
   #pragma omp parallel num_threads(info->tCount)
   {
      // Iterate over data chunk assigned to each thread
      #pragma omp parallel for
      for (int i = 0; i < chunkSize; i++) {
         // Iterate over K cords for each t
         for (int j = 0; j < info->N; j++) {
               // Check distance condition
               if (arePointsInDistance(cords[i].x, cords[i].y, cords[j].x, cords[j].y, info->D)) {
                  // Set flag indicating a point has been found
                  found_count ++;
               }
         }
      }
   }

   // // Communicate results to the master process
   // if (rank != 0) {
   //    // Send the flag to the master process with tag 0
   //    MPI_Send(&found_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
   // } else {
   //    // Receive flags from other processes
   //    for (int i = 1; i < size; i++) {
   //       // Receive the flag with tag i
   //       MPI_Recv(&result_flag, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   //       if (result_flag == 1) {
   //             found_count++;
   //             if (found_count >= 3) {
   //                // Send termination signal to all processes
   //                for (int j = 0; j < size; j++) {
   //                   MPI_Send(&found_flag, 1, MPI_INT, j, size + 1, MPI_COMM_WORLD);
   //                }
   //                break;
   //             }
   //       }
   //    }
   // }

   // Terminate if termination signal received
   // if (found_count >= 3) {
   //    MPI_Finalize();
   //    return 0;
   // }


   MPI_Type_free(&MPI_POINT);
   MPI_Type_free(&MPI_INFO);
   MPI_Finalize();

   return 0;
}


