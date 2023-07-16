#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "myProto.h"
#include "createMPIStruct.h"

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
   Cord *data = NULL;
   Cord *cords = NULL;
   MPI_Datatype MPI_INFO = createInfoStruct();
   MPI_Datatype MPI_CORD = createCordStruct();

   // Read data and exchange information between processes

   clock_t start, end;
   start = clock();
   double exec_time;

   if (rank == 0)
   {
      points = readPointArrayFromFile(argv[1], &info);
      data = initCordsArray(info, points);
      MPI_Send(info, 1, MPI_INFO, 1, 0, MPI_COMM_WORLD);
   }
   else
   {
      info = (Info *)malloc(sizeof(Info));
      MPI_Recv(info, 1, MPI_INFO, 0, 0, MPI_COMM_WORLD, &status);
   }

   int chunkSize = rank == size - 1 ? (int) ceil(info->tCount / size) : info->tCount / size;
   cords = (Cord *) malloc(sizeof(Cord) * chunkSize * info->N);
   MPI_Scatter(data, chunkSize * info->N, MPI_CORD, cords, chunkSize * info->N, MPI_CORD, 0, MPI_COMM_WORLD);

   if (calcCoordinates(cords, info->N, chunkSize) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   // Assuming cords is populated with data
   // Divide the data into chunks

   // OpenMP parallel region

   double *local_results = (double *) malloc(chunkSize * (PCT + 1) * sizeof(double)), *global_results = NULL;
   int local_counter = 0, global_counter = 0;

   #pragma omp parallel num_threads(chunkSize)
   {
      // Iterate over data chunk assigned to each thread
      int count_group = 1;
      int offset = omp_get_thread_num() * info->N;
      double threePoints[PCT + 1] = {0};
      Cord *tCountRegion = cords + offset;
      threePoints[0] = tCountRegion[0].t;

      // Iterate over each region of points per t
      int *satisfiers = calcProximityCriteria(tCountRegion, info->D, info->N, info->K);

      #pragma omp parallel for
      for (int i = 0; i < info->N; i++)
         if (satisfiers[i])
            #pragma omp critical
            {
               if (count_group < PCT + 1)
               {
                  threePoints[count_group] = (double)tCountRegion[i].point.id;
                  count_group++;
               }
            }

      if (count_group == PCT + 1)
      {
         #pragma omp critical
         {
            memcpy(&local_results[local_counter * (PCT + 1)], threePoints, sizeof(threePoints));
            local_counter++;
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if (rank == 0)
      global_results = (double *)malloc(info->tCount * (PCT + 1) * sizeof(double));

   // Communicate results to the master process
   MPI_Gather(local_results, local_counter * (PCT + 1), MPI_DOUBLE, global_results, local_counter * (PCT + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Reduce(&local_counter, &global_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   end = clock(); 
   exec_time = (double) (end - start) / CLOCKS_PER_SEC;

   if (rank == 0)
   {
      if (global_counter == 0)
         printf("There were no 3 points found for any t\n");
      else
      {
         for (int i = 0; i < global_counter; i++)
         {
            int p1 = i * 4 + 1, p2 = i * 4 + 2, p3 = i * 4 + 3, t = i * 4;
            printf("Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", (int)global_results[p1],
                   (int)global_results[p2], (int)global_results[p3], global_results[t]);
         }
      }

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

   points = readPointArrayFromFile(argv[1], &info);
   Cord *cords = (Cord *)malloc(sizeof(Cord) * (info->tCount + 1) * info->N);

   clock_t start, end;
   start = clock();
   double exec_time;

   // Prepare Cords Array
   for (int tCount = 0; tCount <= info->tCount; tCount++)
   {
      double t = 2.0 * tCount / (info->tCount) - 1.0;
      for (int point = 0; point < info->N; point++)
      {
         cords[point + tCount * info->N].point = points[point];
         cords[point + tCount * info->N].t = t;
      }
   }

   // Calculate each Point Cord's per t
   for (int tCount = 0; tCount <= info->tCount; tCount++)
   {
      int offset = tCount * info->N;
      for (int p = 0; p < info->N; p++)
      {
         double x1 = cords[offset + p].point.x1, x2 = cords[offset + p].point.x2, t = cords[offset + p].t;
         double a = cords[offset + p].point.a, b = cords[offset + p].point.b;
         cords[offset + p].x = ((x2 - x1) / 2) * sin(t * PI) + ((x2 + x1) / 2);
         cords[offset + p].y = a * cords[offset + p].x + b;
      }
   }

   // Find Points that satisfy Proximity Criteria
   int count_satisfy = 0;
   double *results = (double *)malloc(4 * (info->tCount + 1) * sizeof(double));
   for (int tCount = 0; tCount <= info->tCount; tCount++)
   {
      double threePoints[4] = {0};
      int offset = tCount * info->N, count_in_distance = 0, count_group = 1;
      Cord *tCountRegion = cords + offset;
      threePoints[0] = tCountRegion[0].t;

      for (int Pi = 0; Pi < info->N && count_group < 4; Pi++)
      {
         for (int Pj = 0; Pj < info->N && count_in_distance < info->K; Pj++)
         {
            if (Pi != Pj)
            {
               double dx = tCountRegion[Pj].x - tCountRegion[Pi].x;
               double dy = tCountRegion[Pj].y - tCountRegion[Pi].y;
               if (sqrt(dx * dx + dy * dy) < info->D)
                  count_in_distance++;
            }
         }

         if (count_in_distance >= info->K)
         {
            threePoints[count_group] = (double)tCountRegion[Pi].point.id;
            count_group++;
         }

         count_in_distance = 0;
      }

      if (count_group == 4)
      {
         memcpy(&results[count_satisfy * 4], threePoints, sizeof(threePoints));
         count_satisfy++;
      }

      count_group = 0;
   }

   end = clock(); 
   exec_time = (double) (end - start) / CLOCKS_PER_SEC;

   if (count_satisfy == 0)
      printf("There were no 3 points found for any t\n");
   else
   {
      for (int i = 0; i < count_satisfy; i++)
      {
         int p1 = i * 4 + 1, p2 = i * 4 + 2, p3 = i * 4 + 3, t = i * 4;
         printf("Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", (int)results[p1],
                (int)results[p2], (int)results[p3], results[t]);
      }
   }

   printf("Execution time: %.6f seconds\n", exec_time);
}
