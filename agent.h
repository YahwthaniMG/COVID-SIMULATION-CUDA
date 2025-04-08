#pragma once
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <direct.h>   // Para _mkdir en Windows
#include <sys/stat.h> // Para stat en Windows
#include <iostream>

// Tipo de simulación
enum SimulationType {
    CPU_SIMULATION,
    GPU_SIMULATION
};

// Estados de infeccion
enum AgentStatus {
    NOT_INFECTED = 0,    // No infectado
    INFECTED = 1,        // Infectado
    QUARANTINED = -1,    // En cuarentena
    DECEASED = -2        // Fallecido
};

// Funcion auxiliar para convertir estado a cadena
inline std::string statusToString(int status) {
    switch (status) {
    case NOT_INFECTED: return "No infectado";
    case INFECTED: return "Infectado";
    case QUARANTINED: return "En cuarentena";
    case DECEASED: return "Fallecido";
    default: return "Desconocido";
    }
}


// Estructura del agente
struct Agent {
    // Posicion
    float x;
    float y;

    // Estado
    int status;

    // Probabilidades individuales
    float p_con;     // Probabilidad de contagio
    float p_ext;     // Probabilidad de contagio externo
    float p_fat;     // Probabilidad de mortalidad
    float p_mov;     // Probabilidad de movilidad
    float p_smo;     // Probabilidad de movilidad corta

    // Tiempos
    int t_inc;       // Tiempo de incubacion
    int t_rec;       // Tiempo de recuperacion

    // Constructor con valores por defecto
    __host__ __device__ Agent() :
        x(0.0f), y(0.0f),
        status(NOT_INFECTED),
        p_con(0.0f), p_ext(0.0f), p_fat(0.0f), p_mov(0.0f), p_smo(0.0f),
        t_inc(0), t_rec(0) {}

};

// Estadisticas diarias
struct DailyStats {
    int infected;           // Nuevos infectados
    int recovered;          // Nuevos recuperados
    int deaths;             // Nuevos fallecidos
    int totalInfected;      // Total acumulado de infectados
    int totalRecovered;     // Total acumulado de recuperados
    int totalDeaths;        // Total acumulado de fallecidos
    int currentInfected;    // Actualmente infectados
    int currentQuarantined; // Actualmente en cuarentena

    DailyStats() : infected(0), recovered(0), deaths(0),
        totalInfected(0), totalRecovered(0), totalDeaths(0),
        currentInfected(0), currentQuarantined(0) {}
};