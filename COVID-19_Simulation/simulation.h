#pragma once
#include "agent.h"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

// Estructura para rastrear movimientos de agentes
struct AgentMovement {
    int day;
    int moveNumber;
    float x;
    float y;
};

// Estructura para rastrear informaciуn de un agente especнfico
struct AgentTracker {
    int id;
    float p_con;      // Probabilidad de contagio
    float p_ext;      // Probabilidad de contagio externo
    float p_fat;      // Probabilidad de mortalidad
    float p_mov;      // Probabilidad de movilidad
    float p_smo;      // Probabilidad de movilidad corta
    std::vector<AgentMovement> movements;
};

class Simulation {
private:
    // Datos de agentes
    Agent* h_agents;            // Agentes en host
    Agent* d_agents;            // Agentes en device
    curandState* d_states;      // Estados aleatorios en device
    int numAgents;              // Nъmero de agentes

    // Estadнsticas
    std::vector<DailyStats> stats;

    // Dнas importantes
    int firstInfectionDay;
    int halfInfectionDay;
    int allInfectionDay;
    int firstRecoveryDay;
    int halfRecoveryDay;
    int allRecoveryDay;
    int firstDeathDay;
    int halfDeathDay;
    int allDeathDay;

    // Seguimiento de agentes
    std::vector<AgentTracker> trackedAgents;
    int numTrackedAgents;

    // Bandera para registro detallado
    bool detailedLogging;

    // Tipo de simulación actual
    SimulationType currentSimType;

    // Mйtodos privados
    void initializeAgents();
    void initializeCUDA();
    void cleanupCUDA();
    void updateStatistics(int day);
    void trackAgentMovements(int day, int moveNumber);

public:
    Simulation();
    ~Simulation();

    void reset();              // Reiniciar simulaciуn
    void runCPU();             // Ejecutar en CPU
    void runGPU();             // Ejecutar en GPU
    void printResults();       // Mostrar resultados
    void printAgentTracking(); // Mostrar seguimiento de agentes

    // Guardar estadísticas a archivos separados para CPU y GPU
    void saveStatsToFile(const std::string& suffix);
};