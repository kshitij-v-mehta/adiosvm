#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    cudaError_t err = cudaGetLastError();\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    printf("Cuda error %d - %s. Exiting..\n", cudaGetErrorString(err));\
    fflush(stdout); fflush(stderr);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    cudaError_t err = cudaGetLastError();\
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    printf("Cuda error %d - %s. Exiting..\n", cudaGetErrorString(err));\
    fflush(stdout); fflush(stderr);\
    return EXIT_FAILURE;}} while(0)

int curand_init_gen(double *devdata, int n, curandGenerator_t *gen);
int gen_curand_values (double *devdata, int n, curandGenerator_t gen);
int curand_cleanup(curandGenerator_t gen);

