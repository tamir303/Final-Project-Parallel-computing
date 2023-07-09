#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "myProto.h"
#include "mathCalc.h"
#include "createMPIStruct.h"

#define PCT 3 // Number of points to satisfy the proximity criteria

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

   if (calcCoordinates(cords, info->N, chunkSize) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   MPI_Barrier(MPI_COMM_WORLD);

   for (int i = 0; i < info->tCount / size * info->N; i++)
      printf("proc: %d, pointID: %d, t: %lf, x: %lf, y: %lf\n", rank, cords[i].point.id, cords[i].t, cords[i].x, cords[i].y);

   // Assuming cords is populated with data
   // Divide the data into equal-sized chunks

   // OpenMP parallel region
   char** results[chunkSize];
   int countResults = 0;
   #pragma omp parallel num_threads(chunkSize)
   {
      // Iterate over data chunk assigned to each thread
      int found_point = 0;
      int found_group = 0;
      #pragma omp parallel for
      for (int t = 0; t < chunkSize; t++) {
         int offset = t * info->N;
         int threePoints = {0, 0 ,0}
         Cord* tCountRegion = cords + offset;
         // Iterate over K cords for each t
         for (int Pi = 0; Pi < info->N && found_group != PCT; Pi++) {
            for (int Pj = Pi + 1; Pj < info->N && found_point != info->K; Pj++)
               // Check distance condition
               if (arePointsInDistance(tCountRegion[Pi].x, tCountRegion[Pi].y, tCountRegion[Pj].x, tCountRegion[Pj].y, info->D)) {
                  // Set flag indicating a point has been found
                  found_point ++;
               }

            if (found_point == info->K) {
               threePoints[found_group ++] = tCountRegion[Pi].point.id;
               found_point = 0;
            }
         }

         if (found_group == PCT) {
            results[countResults][100];
            sprintf(results[countResults], "Points %d, %d, %d satisfy Proximity Criteria at t = %lf",
               threePoints[0], threePoints[1], threePoints[2], tCountRegion[Pi].t);
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

   MPI_Type_free(&MPI_INFO);
   MPI_Type_free(&MPI_CORD);
   MPI_Finalize();

   return 0;
}


