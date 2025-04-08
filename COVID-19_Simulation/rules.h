#pragma once
#include "agent.h"
#include <curand_kernel.h>

// Regla 1: Contagio
__global__ void applyRule1Kernel(Agent* agents, int n, curandState* states, float contagionRadius);
void applyRule1CPU(Agent* agents, int n, float contagionRadius);

// Regla 2: Movilidad
__global__ void applyRule2Kernel(Agent* agents, int n, curandState* states, float maxLocalRadius, float areaWidth, float areaHeight);
void applyRule2CPU(Agent* agents, int n, float maxLocalRadius, float areaWidth, float areaHeight);

// Regla 3: Contagio externo
__global__ void applyRule3Kernel(Agent* agents, int n, curandState* states);
void applyRule3CPU(Agent* agents, int n);

// Regla 4: Tiempo de incubación, cuarentena y recuperación
__global__ void applyRule4Kernel(Agent* agents, int n);
void applyRule4CPU(Agent* agents, int n);

// Regla 5: Casos fatales
__global__ void applyRule5Kernel(Agent* agents, int n, curandState* states);
void applyRule5CPU(Agent* agents, int n);