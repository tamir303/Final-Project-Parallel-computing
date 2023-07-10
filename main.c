#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "myProto.h"
#include "createMPIStruct.h"

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
   Results* res = NULL;
   Results* allRes = NULL;
   MPI_Datatype MPI_INFO = createInfoStruct();
   MPI_Datatype MPI_CORD = createCordStruct();
   MPI_Datatype MPI_RES = createResultStruct();

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

   // for (int i = 0; i < info->tCount / size * info->N; i++)
   //    printf("proc: %d, pointID: %d, t: %lf, x: %lf, y: %lf\n", rank, cords[i].point.id, cords[i].t, cords[i].x, cords[i].y);

   // Assuming cords is populated with data
   // Divide the data into equal-sized chunks

   // OpenMP parallel region
   res = (Results*) malloc(sizeof(Results));
   res->results = (char**) malloc(chunkSize * sizeof(char*));
   res->resultCounter = 0;
   #pragma omp parallel num_threads(chunkSize)
   {
      // Iterate over data chunk assigned to each thread
      int count_group = 0;
      int offset = omp_get_thread_num() * info->N;
      int threePoints[PCT] = { 0 };
      Cord* tCountRegion = cords + offset;

      // Iterate over each region of points per t
      int* satisfiers = calcProximityCriteria(tCountRegion, info->D, info->N, info->K);

      #pragma omp parallel for
      for (int i = 0; i < info->N; i++)
         if (satisfiers[i]) {
            #pragma omp critical
            {
               if (count_group < PCT) {
                  threePoints[count_group] = tCountRegion[i].point.id;
                  count_group ++;
               }
            }
         }

      if (count_group == PCT) {
         #pragma omp critical 
         {
            res->results[res->resultCounter] = (char*) malloc(100);
            char str[100];
            sprintf(str, "Points %d, %d, %d satisfy Proximity Criteria at t = %lf",
               threePoints[0], threePoints[1], threePoints[2], tCountRegion[0].t);
            strcpy(res->results[res->resultCounter], str);
            res->resultCounter++;
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // Communicate results to the master process
   if (rank == 0)
      allRes = (Results*) malloc(size * sizeof(Results));
   MPI_Gather(res, 1, MPI_RES, allRes, 1, MPI_RES, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      int count = 0;
      for (int i = 0; i < size; i++) {
         for (int j = 0; j < allRes[i].resultCounter; j++)
            printf("%s\n",allRes[i].results[j]);
         printf("%d", allRes[i].resultCounter);
         count += allRes[i].resultCounter;
      }

      if (count == 0)
         printf("There were no 3 points found for any t\n");
   }

   MPI_Type_free(&MPI_INFO);
   MPI_Type_free(&MPI_CORD);
   MPI_Type_free(&MPI_RES);
   MPI_Finalize();

   return 0;
}


