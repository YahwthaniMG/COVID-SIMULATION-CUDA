#include <iostream>
#include <chrono>
#include "simulation.h"

int main() {
    std::cout << "COVID-19 Infection Model Analysis" << std::endl;
    std::cout << "=================================" << std::endl;

    // Crear instancia de simulación
    Simulation simulation;

    // Ejecutar versión CPU
    std::cout << "\nRunning CPU version..." << std::endl;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    simulation.runCPU();
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;

    // Mostrar resultados CPU
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " seconds" << std::endl;
    simulation.printResults();

    // Reiniciar simulación
    simulation.reset();

    // Ejecutar versión GPU
    std::cout << "\nRunning GPU version..." << std::endl;
    auto gpuStart = std::chrono::high_resolution_clock::now();
    simulation.runGPU();
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuDuration = gpuEnd - gpuStart;

    // Mostrar resultados GPU
    std::cout << "GPU Execution Time: " << gpuDuration.count() << " seconds" << std::endl;
    simulation.printResults();

    // Mostrar speedup
    double speedup = cpuDuration.count() / gpuDuration.count();
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

    // Informar sobre la ubicación de los archivos
    std::cout << "\nStatistics files have been saved:" << std::endl;
    std::cout << "- CPU simulation: daily_stats_cpu.csv" << std::endl;
    std::cout << "- GPU simulation: daily_stats_gpu.csv" << std::endl;

    return 0;
}