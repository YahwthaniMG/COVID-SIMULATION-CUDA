#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// Calcula la distancia euclidiana entre dos puntos
__host__ __device__ inline float calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// Inicializar estados aleatorios para CUDA
__global__ void initRandomStates(curandState* states, unsigned long seed, int n);

// Verificar errores CUDA
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}