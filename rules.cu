#define _USE_MATH_DEFINES
#include "rules.h"
#include "utils.h"
#include "config.h"
#include <random>
#include <cmath>
#include <device_launch_parameters.h>

// Definir PI si no está disponible
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// IMPLEMENTACIÓN DE REGLAS EN GPU ======================================================

// Regla 1: Contagio
__global__ void applyRule1Kernel(Agent* agents, int n, curandState* states, float contagionRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Solo procesamos agentes sanos
    if (agents[idx].status != NOT_INFECTED) return;

    // Buscar agentes infectados cercanos
    bool hasInfectedNearby = false;

    for (int j = 0; j < n; j++) {
        if (j == idx) continue;

        // Si el otro agente está infectado (no en cuarentena)
        if (agents[j].status == INFECTED) {
            float distance = calculateDistance(agents[idx].x, agents[idx].y, agents[j].x, agents[j].y);

            // Si está dentro del radio de contagio
            if (distance <= contagionRadius) {
                hasInfectedNearby = true;
                break;
            }
        }
    }

    // Si hay un infectado cerca, verificar probabilidad de contagio
    if (hasInfectedNearby) {
        float random = curand_uniform(&states[idx]);
        if (random <= agents[idx].p_con) {
            agents[idx].status = INFECTED;
        }
    }
}

// Regla 2: Movilidad
__global__ void applyRule2Kernel(Agent* agents, int n, curandState* states, float maxLocalRadius, float areaWidth, float areaHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Agentes en cuarentena o fallecidos no se mueven
    if (agents[idx].status == QUARANTINED || agents[idx].status == DECEASED) return;

    // Determinar si el agente se mueve según su probabilidad de movilidad
    float moveRand = curand_uniform(&states[idx]);
    if (moveRand <= agents[idx].p_mov) {

        // Determinar si el movimiento es local o distante
        float shortMoveRand = curand_uniform(&states[idx]);
        bool isLocalMove = shortMoveRand <= agents[idx].p_smo;

        if (isLocalMove) {
            // Movimiento local en un radio cercano
            float angle = 2.0f * (float)M_PI * curand_uniform(&states[idx]);
            float radius = maxLocalRadius * curand_uniform(&states[idx]);

            agents[idx].x += radius * cosf(angle);
            agents[idx].y += radius * sinf(angle);
        }
        else {
            // Movimiento a cualquier punto del área
            agents[idx].x = areaWidth * curand_uniform(&states[idx]);
            agents[idx].y = areaHeight * curand_uniform(&states[idx]);
        }

        // Asegurar que el agente permanezca dentro de los límites
        agents[idx].x = fmaxf(0.0f, fminf(agents[idx].x, areaWidth));
        agents[idx].y = fmaxf(0.0f, fminf(agents[idx].y, areaHeight));
    }
}

// Regla 3: Contagio externo
__global__ void applyRule3Kernel(Agent* agents, int n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Solo aplicamos a agentes sanos
    if (agents[idx].status != NOT_INFECTED) return;

    // Verificar probabilidad de contagio externo
    float random = curand_uniform(&states[idx]);
    if (random <= agents[idx].p_ext) {
        agents[idx].status = INFECTED;
    }
}

// Regla 4: Tiempo de incubación, cuarentena y recuperación
__global__ void applyRule4Kernel(Agent* agents, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Si el agente está infectado (no en cuarentena)
    if (agents[idx].status == INFECTED) {
        // Reducir tiempo de incubación
        agents[idx].t_inc--;

        // Si el tiempo de incubación terminó, pasar a cuarentena
        if (agents[idx].t_inc <= 0) {
            agents[idx].status = QUARANTINED;
            agents[idx].t_rec = RECOVERY_TIME;
        }
    }
    // Si el agente está en cuarentena
    else if (agents[idx].status == QUARANTINED) {
        // Reducir tiempo de recuperación
        agents[idx].t_rec--;

        // Si el tiempo de recuperación terminó, volver a sano
        if (agents[idx].t_rec <= 0) {
            agents[idx].status = NOT_INFECTED;
        }
    }
}

// Regla 5: Casos fatales
__global__ void applyRule5Kernel(Agent* agents, int n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Solo aplicamos a agentes en cuarentena
    if (agents[idx].status != QUARANTINED) return;

    // Verificar probabilidad de fatalidad
    float random = curand_uniform(&states[idx]);
    if (random <= agents[idx].p_fat) {
        agents[idx].status = DECEASED;
    }
}

// IMPLEMENTACIÓN DE REGLAS EN CPU ======================================================

// Regla 1: Contagio
void applyRule1CPU(Agent* agents, int n, float contagionRadius) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        // Solo procesamos agentes sanos
        if (agents[i].status != NOT_INFECTED) continue;

        // Buscar agentes infectados cercanos
        bool hasInfectedNearby = false;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;

            // Si el otro agente está infectado (no en cuarentena)
            if (agents[j].status == INFECTED) {
                float distance = calculateDistance(agents[i].x, agents[i].y, agents[j].x, agents[j].y);

                // Si está dentro del radio de contagio
                if (distance <= contagionRadius) {
                    hasInfectedNearby = true;
                    break;
                }
            }
        }

        // Si hay un infectado cerca, verificar probabilidad de contagio
        if (hasInfectedNearby) {
            float random = dis(gen);
            if (random <= agents[i].p_con) {
                agents[i].status = INFECTED;
            }
        }
    }
}

// Regla 2: Movilidad
void applyRule2CPU(Agent* agents, int n, float maxLocalRadius, float areaWidth, float areaHeight) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        // Agentes en cuarentena o fallecidos no se mueven
        if (agents[i].status == QUARANTINED || agents[i].status == DECEASED) continue;

        // Determinar si el agente se mueve según su probabilidad de movilidad
        float moveRand = dis(gen);
        if (moveRand <= agents[i].p_mov) {

            // Determinar si el movimiento es local o distante
            float shortMoveRand = dis(gen);
            bool isLocalMove = shortMoveRand <= agents[i].p_smo;

            if (isLocalMove) {
                // Movimiento local en un radio cercano
                float angle = 2.0f * (float)M_PI * dis(gen);
                float radius = maxLocalRadius * dis(gen);

                agents[i].x += radius * cosf(angle);
                agents[i].y += radius * sinf(angle);
            }
            else {
                // Movimiento a cualquier punto del área
                agents[i].x = areaWidth * dis(gen);
                agents[i].y = areaHeight * dis(gen);
            }

            // Asegurar que el agente permanezca dentro de los límites
            agents[i].x = std::max(0.0f, std::min(agents[i].x, areaWidth));
            agents[i].y = std::max(0.0f, std::min(agents[i].y, areaHeight));
        }
    }
}

// Regla 3: Contagio externo
void applyRule3CPU(Agent* agents, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        // Solo aplicamos a agentes sanos
        if (agents[i].status != NOT_INFECTED) continue;

        // Verificar probabilidad de contagio externo
        float random = dis(gen);
        if (random <= agents[i].p_ext) {
            agents[i].status = INFECTED;
        }
    }
}

// Regla 4: Tiempo de incubación, cuarentena y recuperación
void applyRule4CPU(Agent* agents, int n) {
    for (int i = 0; i < n; i++) {
        // Si el agente está infectado (no en cuarentena)
        if (agents[i].status == INFECTED) {
            // Reducir tiempo de incubación
            agents[i].t_inc--;

            // Si el tiempo de incubación terminó, pasar a cuarentena
            if (agents[i].t_inc <= 0) {
                agents[i].status = QUARANTINED;
                agents[i].t_rec = RECOVERY_TIME;
            }
        }
        // Si el agente está en cuarentena
        else if (agents[i].status == QUARANTINED) {
            // Reducir tiempo de recuperación
            agents[i].t_rec--;

            // Si el tiempo de recuperación terminó, volver a sano
            if (agents[i].t_rec <= 0) {
                agents[i].status = NOT_INFECTED;
            }
        }
    }
}

// Regla 5: Casos fatales
void applyRule5CPU(Agent* agents, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        // Solo aplicamos a agentes en cuarentena
        if (agents[i].status != QUARANTINED) continue;

        // Verificar probabilidad de fatalidad
        float random = dis(gen);
        if (random <= agents[i].p_fat) {
            agents[i].status = DECEASED;
        }
    }
}