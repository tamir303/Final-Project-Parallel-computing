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

   // Assuming cords is populated with data
   // Divide the data into chunks

   // OpenMP parallel region

   double **local_results = (double**) malloc(chunkSize * sizeof(double*)), (*global_results)[4] = NULL;
   int local_counter = 0, global_counter = 0;

   #pragma omp parallel num_threads(chunkSize)
   {
      // Iterate over data chunk assigned to each thread
      int count_group = 1;
      int offset = omp_get_thread_num() * info->N;
      double threePoints[PCT + 1] = { 0 };
      Cord* tCountRegion = cords + offset;
      threePoints[0] = tCountRegion[0].t;

      // Iterate over each region of points per t
      int* satisfiers = calcProximityCriteria(tCountRegion, info->D, info->N, info->K);

      #pragma omp parallel for
      for (int i = 0; i < info->N; i++)
         if (satisfiers[i])
            #pragma omp critical
            {
               if (count_group < PCT + 1) {
                  threePoints[count_group] = (double) tCountRegion[i].point.id;
                  count_group ++;
               }
            }

      if (count_group == PCT + 1) {
         #pragma omp critical 
         {
            local_results[local_counter] = (double*) malloc((PCT + 1) * sizeof(double));
            local_results[local_counter][0] = threePoints[0];
            local_results[local_counter][1] = threePoints[1];
            local_results[local_counter][2] = threePoints[2];
            local_results[local_counter][3] = threePoints[3];
            local_counter ++;
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if (rank == 0)
      global_results = (double (*)[4]) malloc(info->tCount * sizeof(double[4]));

   // Communicate results to the master process
   MPI_Gather(local_results, local_counter * (PCT + 1), MPI_DOUBLE, global_results, local_counter * (PCT + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Reduce(&local_counter, &global_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   // if (rank == 0) {
   //    if (global_counter == 0)
   //       printf("There were no 3 points found for any t\n");
   //    else
   //       while (global_counter >= 0) {
   //          global_counter --;
   //          printf("Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", (int) global_results[global_counter][1], (int) global_results[global_counter][2]
   //             ,(int) global_results[global_counter][3], global_results[global_counter][4]);
   //       }
   // }

   MPI_Type_free(&MPI_INFO);
   MPI_Type_free(&MPI_CORD);
   MPI_Finalize();

   return 0;
}


