/**
 * This program computes the brownian motion equation
 * in 1D using the CUDA interface for NVIDIA GPUs.
 * 
 * Written by: Gavin Wale
 *             ME471: Parallel Scientific Computing
 *             Boise State University
 *             5/1/2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 8 // to be adjusted up to 1024


__device__ float brownian(float sigma, int N, float dt);
__device__ float normal_dist();



__global__ void brownianOnGPU_globalAtomic(int n, float *dx_max, float *dx_min, float *dx_mean, float *dx_std) {

    float sigma = 0.1;
    int N = 100;
    float dt = 0.1;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {

        float x = brownian(sigma, N, dt); // Every thread calculates x

        __syncthreads(); // Avoid data race
        if(x>*dx_max) *dx_max = x; // If x is > dx_max, new dx_max

        __syncthreads(); // Avoid data race
        if(x<*dx_min) *dx_min = x; // If x is < dx_min, new dx_min


        // atomicAdd(*dx_mean, x);
        __syncthreads();
        *dx_mean = *dx_mean + x;
        
        __syncthreads();
        
        float temp = x*x;
        
        *dx_std = *dx_std + temp;
        
        //atomicAdd(*dx_std,x*x);
        
    }
}

/**
 * Device function that can be called only by the GPU.
 * Returns the calculation of a brownian and makes an internal call
 * to another device function.
 * 
 * @param sigma 
 * @param N
 * @param dt 
 * @return x - a float representing brownian motion 
 */
__device__ float brownian(float sigma, int N, float dt) {
    float dW, mu;
    float x = 1; //initial value for the brownian motion experiment

    for (int i=1; i<N; i++){

        mu = sin(i*dt + M_PI/4); // mu
        dW = sqrt(dt)*normal_dist(); // noise
        x = x + mu*x*dt + sigma*x*dW;
    }
    return x;
}

/**
 * Device function to create noise for x.
 * Called in the device function "brownian".
 * 
 * @return x - a float with noise
 */
__device__ float normal_dist() {

    int id = threadIdx.x;
    curandState state;
    curand_init ( id + 1, id, 0, &state);
    float U1 = curand_uniform( &state);
    float U2 = curand_uniform( &state);

    float x = sqrt(-2*log(U1))*cos(2*M_PI*U2);
    return x;
    
}


/**
 * Main function, calls all necessary functions
 * to calculate the statistics of the brownian motion
 */
int main() {

    // Timer variables
    double iStart, iElaps;
    
    // Set up GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int N_experiments = 10000000; // 10^7 experiments

    // Allocate memory on host
    float *hx_max, *hx_min, *hx_mean, *hx_std;
    hx_max = (float *)malloc(sizeof(float));
    hx_min = (float *)malloc(sizeof(float));
    hx_mean = (float *)malloc(sizeof(float));
    hx_std = (float *)malloc(sizeof(float));
    //hx_max = 0;
    //hx_min = 1e7;
    //hx_mean = 0;
    //hx_std = 0; 

    // Allocate memoroy on device
    float *dx_max, *dx_min, *dx_mean, *dx_std;
    CHECK(cudaMalloc((float **)&dx_max, sizeof(float)));
    CHECK(cudaMalloc((float **)&dx_min, sizeof(float)));
    CHECK(cudaMalloc((float **)&dx_mean, sizeof(float)));
    CHECK(cudaMalloc((float **)&dx_std, sizeof(float)));

    // Set each value to what they should be on the device
    CHECK(cudaMemset(dx_max, 0, sizeof(float)));
    CHECK(cudaMemset(dx_min, 1e7, sizeof(float)));
    CHECK(cudaMemset(dx_mean, 0, sizeof(float)));
    CHECK(cudaMemset(dx_std, 0, sizeof(float)));

    //CHECK(cudaMemcpy(dx_max, hx_max, sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(dx_min, hx_min, sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(dx_mean, hx_mean, sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(dx_std, hx_std, sizeof(float), cudaMemcpyHostToDevice));

    // Number of threads and blocks to call below
    int nThreads = 8;
    int nBlocks = ((10000000 + 8 - 1) / 8);

    // Device runs brownian with input number of blocks and threads for N_experiments
    
    // Timer start
    iStart = seconds();
    
    brownianOnGPU_globalAtomic<<<nBlocks, nThreads>>>(N_experiments, dx_max, dx_min, dx_mean, dx_std);
    CHECK(cudaDeviceSynchronize()); // No clue what this does

    // Print time it took to run brownianOnGPU_globalAtomic for specific threads, blocks
    // printf("brownianOnGPU_globalAtomic <<< %d, %d >>> Time elapsed: sec\n", nblocks, nThreads);

    // Check kernel error
    CHECK(cudaGetLastError());

    // Send results from device to host
    CHECK(cudaMemcpy(&hx_max, dx_max, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hx_min, dx_min, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hx_mean, dx_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hx_std, dx_std, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Timer stop
    iElaps = seconds() - iStart;

    // Calculating actual means and standard deviation off-device
    *hx_mean = *hx_mean / N_experiments;
    *hx_std = sqrt(*hx_std / N_experiments - *hx_mean * *hx_mean);

    // Print results to console on host
    printf("GPU time: %lf\n",iElaps);
    printf("Threads: %d   Blocks: %d\n",nThreads,nBlocks);
    printf("Global mean: %lf\n",hx_mean);
    printf("Global standard deviation: %lf\n",hx_std);
    printf("Global maximum: %lf\n",hx_max);
    printf("Global minimum: %lf\n",hx_min);

    // Free device global memory
    CHECK(cudaFree(dx_max));
    CHECK(cudaFree(dx_min));
    CHECK(cudaFree(dx_mean));
    CHECK(cudaFree(dx_std));

    // Free host memory
    free(hx_max);
    free(hx_min);
    free(hx_mean);
    free(hx_std);

    return(0);

}








