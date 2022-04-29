/***************************
This program computes the brownian motion equation in 1D

Written by: Michal A. Kopera
            Department of Mathematics
            Boise State University
            4/15/2022

 **************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


float normal_dist();
float brownian(float sigma, int N, float dt);
int brownian_history(float *x, float sigma, int N, float dt);

void main(){

  float sigma = 0.1;
  float dt = 0.1;
  int N = 100;

  //**** THIS PART OF THE CODE COMPUTES THE TIME HISTORY OF A GEOMETRIC BROWNIAN MOTION EQUATION - REMOVE IF ONLY WANT TO COMPUTE THE FINAL STATE
  
  //Allocate memory for the evolution of equation (if needed, remove if just computing the final state)
  float *x_history;
  x_history = (float *)malloc(N*sizeof(float));

  time_t t;
  srand((unsigned) time(&t)); //initialize the random seed

  brownian_history(x_history, sigma, N, dt);

  //print fo screen
  for(int j=0; j<N; j++){
    printf("%f, %f\n",j*dt,x_history[j]);
  }
  
  
  //**** THE FOLLOWING SECTION COMPUTES 100 VALUES OF A FINAL STATE OF THE GEOMETRIC BROWNIAN MOTION EQUATION - UNCOMMENT IF COMPUTING THE STATISTICS

  float x_max = 0;
  float x_min = 1e7;
  float x_mean = 0;
  float x_std = 0;
  int N_experiments = 10000;
  for(int j=0; j<N_experiments; j++){
    float x = brownian(sigma, N, dt);
    //printf("%f\n",x);
    if(x>x_max) x_max = x;
    if(x<x_min) x_min = x;
    x_mean = x_mean + x;
    x_std = x_std + x*x;
  }
  x_mean = x_mean/N_experiments;
  x_std = sqrt(x_std/N_experiments - x_mean*x_mean);
  printf("x_max = %f, x_min = %f, x_mean = %f. x_std = %f\n",x_max,x_min,x_mean,x_std);
  
}


float  normal_dist(){

  float U1 = (float)rand() / (float)RAND_MAX;
  float U2 = (float)rand() / (float)RAND_MAX;
  
  float x = sqrt(-2*log(U1))*cos(2*M_PI*U2);
  //printf("%f, %f\n",U1,U2);
  return x;
}

float brownian(float sigma, int N, float dt){
  // This subroutine computes the state of the brownian motion simulation at time N*dt without storing the entire history of x
  float dW, mu;
  float x = 1; //initial value for the brownian motion experiment

  for (int i=1; i<N; i++){
    mu = sin(i*dt + M_PI/4);
    dW = sqrt(dt)*normal_dist();
    x = x + mu*x*dt + sigma*x*dW;
    //printf("%f, ",x);
  }
  //printf("\n");

  return x;

}

int brownian_history(float *x, float sigma, int N, float dt){
  // This subroutine computes the evolution of the equation for geometric Brownian motion and stores it in array x
  
  float dW, mu;
  float x0 = 1; //initial value for the brownian motion experiment
  x[0] = x0;
  for (int i=1; i<N; i++){
    mu = sin(i*dt + M_PI/4);
    dW = sqrt(dt)*normal_dist();
    x[i] = x[i-1] + mu*x[i-1]*dt + sigma*x[i-1]*dW;
  }

  return 0;

}

  
