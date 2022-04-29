/***************************
This program computes the brownian motion equation in 1D

Written by: Gavin Wale
            ME471: Parallel Scientific Computing
            Boise State University
            4/27/2022

 **************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>


float normal_dist();
float brownian(float sigma, int N, float dt);
int brownian_history(float *x, float sigma, int N, float dt);

void main(int argc, char* argv[]) {

    // Initialize as 0, set in following functions
    int nproc, irank = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Check communicator size
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    
    // Check rank number 
    MPI_Comm_rank(MPI_COMM_WORLD,&irank);

double startTotalTime = MPI_Wtime(); // Start of main function time

    // Variables for the experiment
    float sigma = 0.1;
    float dt = 0.1;
    int N = 100;
    float x_max = 0;
    float x_min = 1e7;
    float x_mean = 0;
    float x_std = 0;
    int N_experiments = 10000000; // 10^7 experiments

    // Each processor will run the experiment N_loc times
    int N_loc = N_experiments/nproc;

double startCompTime = MPI_Wtime(); // Start of computation time
    
    for(int j=0; j<N_loc; j++){ // For every value in N_loc

        float x = brownian(sigma, N, dt); // Compute the brownian at x

        if(x>x_max) x_max = x; // If x is bigger than the previous x_max..
        if(x<x_min) x_min = x; // If x is smaller than the previous x_min..
        x_mean = x_mean + x; // Add x to the x_mean variable
        x_std = x_std + x*x; // Add x^2 to the x_std variable
    }

    x_mean = x_mean/N_loc; // Computes the mean x
    x_std = sqrt(x_std/N_loc - x_mean*x_mean); // Computes the stdev of x

double endCompTime = MPI_Wtime(); // End of computatoin time

    // Global variables for MPI_Reduce
    float globalMeanSum;
    float globalMax;
    float globalMin;
    float globalStdevSum;


double startCommTime = MPI_Wtime(); // Beginning of communcation time

    // Communication between processors
    MPI_Reduce(&x_mean, &globalMeanSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // Sums all means
    MPI_Reduce(&x_max, &globalMax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD); // Sets global max value to globalMax
    MPI_Reduce(&x_min, &globalMin, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD); // Sets global min value to globalMin
    MPI_Reduce(&x_std, &globalStdevSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // Sums all stdevs

double endCommTime = MPI_Wtime(); // End of communcation time

    // Calculation of global mean and stdevs
    float globalMean = globalMeanSum / nproc;
    float globalStdev = globalStdevSum / nproc;

double endTotalTime = MPI_Wtime(); // End of main function time



/**
 * This communication is not considered in the calculation as it has no effect
 * on the brownian motion and is purely for analysis purposes.
 */
double totalTimeLocal = endTotalTime - startTotalTime;
double commTimeLocal = endCommTime - startCommTime;
double compTimeLocal = endCompTime - startCompTime;

double totalTimeSum;
double commTimeSum;
double compTimeSum;

MPI_Reduce(&totalTimeLocal, &totalTimeSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Total time reduction
MPI_Reduce(&commTimeLocal, &commTimeSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Communication time reduction
MPI_Reduce(&compTimeLocal, &compTimeSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Computation time reduction

double totalTime = totalTimeSum / nproc;
double commTime = commTimeSum / nproc;
double compTime = compTimeSum / nproc;

double percentageIsComm = (commTime / totalTime) * 100;



    if (irank == 0) {

	printf("N: %d\n",N_experiments);
	printf("nproc: %d\n",nproc);
        printf("Global mean: %lf\n",globalMean);
        printf("Global standard deviation: %lf\n",globalStdev);
        printf("Global maximum: %lf\n",globalMax);
        printf("Global minimum: %lf\n",globalMin);
        printf("Communication time: %lf\n",commTime);
        printf("Calculation time: %lf\n",compTime);
        printf("Total time: %lf\n",totalTime);
        printf("Percentage of execution time communicating: %lf\n",percentageIsComm);

    }


    // Close MPI interface
    MPI_Finalize();

}

/**
 * Provides the noise for the brownian function below.
 * Makes the randomness more natural.
 * 
 * @return float 
 */
float  normal_dist(){
  float U1 = (float)rand() / (float)RAND_MAX;
  float U2 = (float)rand() / (float)RAND_MAX;
  float x = sqrt(-2*log(U1))*cos(2*M_PI*U2);
  return x;
}

/**
 * Computes the brownian with given parameters.
 * Used in a loop in the main function.
 * 
 * @param sigma - the stochasatic element
 * @param N - the number of runs
 * @param dt - the time interval
 * 
 * @return x - The x value at the given time in brownian motion
 */
float brownian(float sigma, int N, float dt){
  float dW, mu;
  float x = 1; //initial value for the brownian motion experiment
  for (int i=1; i<N; i++){
    mu = sin(i*dt + M_PI/4);
    dW = sqrt(dt)*normal_dist();
    x = x + mu*x*dt + sigma*x*dW;
  }
  return x;
}

  
