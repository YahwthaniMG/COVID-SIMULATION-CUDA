#include "simulation.h"
#include "config.h"
#include "utils.h"
#include "rules.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <device_launch_parameters.h>

Simulation::Simulation() :
    h_agents(nullptr),
    d_agents(nullptr),
    d_states(nullptr),
    numAgents(N),
    firstInfectionDay(-1),
    halfInfectionDay(-1),
    allInfectionDay(-1),
    firstRecoveryDay(-1),
    halfRecoveryDay(-1),
    allRecoveryDay(-1),
    firstDeathDay(-1),
    halfDeathDay(-1),
    allDeathDay(-1),
    numTrackedAgents(5), // Rastreamos 5 agentes como muestra
    currentSimType(CPU_SIMULATION) // Por defecto empezamos con CPU
{
    // Inicialización
    h_agents = new Agent[numAgents];
    initializeAgents();

    // Inicializar agentes a rastrear
    trackedAgents.clear();
    for (int i = 0; i < numTrackedAgents; i++) {
        AgentTracker tracker;
        tracker.id = i; // Rastreamos los primeros numTrackedAgents
        tracker.p_con = h_agents[i].p_con;
        tracker.p_ext = h_agents[i].p_ext;
        tracker.p_fat = h_agents[i].p_fat;
        tracker.p_mov = h_agents[i].p_mov;
        tracker.p_smo = h_agents[i].p_smo;
        trackedAgents.push_back(tracker);

        // Registrar posición inicial
        AgentMovement initialPos;
        initialPos.day = 0;
        initialPos.moveNumber = 0;
        initialPos.x = h_agents[i].x;
        initialPos.y = h_agents[i].y;
        trackedAgents[i].movements.push_back(initialPos);
    }
}

Simulation::~Simulation() {
    if (h_agents) {
        delete[] h_agents;
    }
    cleanupCUDA();
}

void Simulation::reset() {
    // Limpiar estadísticas
    stats.clear();

    // Reiniciar días importantes
    firstInfectionDay = -1;
    halfInfectionDay = -1;
    allInfectionDay = -1;
    firstRecoveryDay = -1;
    halfRecoveryDay = -1;
    allRecoveryDay = -1;
    firstDeathDay = -1;
    halfDeathDay = -1;
    allDeathDay = -1;

    // Reiniciar agentes
    initializeAgents();

    // Reiniciar seguimiento
    trackedAgents.clear();
    for (int i = 0; i < numTrackedAgents; i++) {
        AgentTracker tracker;
        tracker.id = i;
        tracker.p_con = h_agents[i].p_con;
        tracker.p_ext = h_agents[i].p_ext;
        tracker.p_fat = h_agents[i].p_fat;
        tracker.p_mov = h_agents[i].p_mov;
        tracker.p_smo = h_agents[i].p_smo;
        trackedAgents.push_back(tracker);

        // Registrar posición inicial
        AgentMovement initialPos;
        initialPos.day = 0;
        initialPos.moveNumber = 0;
        initialPos.x = h_agents[i].x;
        initialPos.y = h_agents[i].y;
        trackedAgents[i].movements.push_back(initialPos);
    }
}

void Simulation::initializeAgents() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> positionDis(0.0f, 1.0f);
    std::uniform_real_distribution<float> contagionDis(MIN_CONTAGION_PROB, MAX_CONTAGION_PROB);
    std::uniform_real_distribution<float> externalDis(MIN_EXTERNAL_PROB, MAX_EXTERNAL_PROB);
    std::uniform_real_distribution<float> mortalityDis(MIN_MORTALITY_PROB, MAX_MORTALITY_PROB);
    std::uniform_real_distribution<float> mobilityDis(MIN_MOBILITY_PROB, MAX_MOBILITY_PROB);
    std::uniform_real_distribution<float> shortMobilityDis(MIN_SHORT_MOBILITY_PROB, MAX_SHORT_MOBILITY_PROB);
    std::uniform_int_distribution<int> incubationDis(MIN_INCUBATION_TIME, MAX_INCUBATION_TIME);

    for (int i = 0; i < numAgents; i++) {
        // Posición inicial aleatoria
        h_agents[i].x = AREA_WIDTH * positionDis(gen);
        h_agents[i].y = AREA_HEIGHT * positionDis(gen);

        // Estado inicial (no infectado)
        h_agents[i].status = NOT_INFECTED;

        // Probabilidades individuales
        h_agents[i].p_con = contagionDis(gen);
        h_agents[i].p_ext = externalDis(gen);
        h_agents[i].p_fat = mortalityDis(gen);
        h_agents[i].p_mov = mobilityDis(gen);
        h_agents[i].p_smo = shortMobilityDis(gen);

        // Tiempos
        h_agents[i].t_inc = incubationDis(gen);
        h_agents[i].t_rec = RECOVERY_TIME;
    }
}

void Simulation::initializeCUDA() {
    // Alocar memoria en GPU para agentes
    cudaMalloc(&d_agents, numAgents * sizeof(Agent));
    cudaCheckError();

    // Copiar agentes de CPU a GPU
    cudaMemcpy(d_agents, h_agents, numAgents * sizeof(Agent), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Alocar memoria para estados aleatorios
    cudaMalloc(&d_states, numAgents * sizeof(curandState));
    cudaCheckError();

    // Inicializar estados aleatorios
    int blocks = (numAgents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandomStates << <blocks, BLOCK_SIZE >> > (d_states, static_cast<unsigned long>(time(NULL)), numAgents);
    cudaCheckError();
}

void Simulation::cleanupCUDA() {
    if (d_agents) {
        cudaFree(d_agents);
        d_agents = nullptr;
    }

    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}

void Simulation::trackAgentMovements(int day, int moveNumber) {
    for (int i = 0; i < numTrackedAgents && i < numAgents; i++) {
        AgentMovement movement;
        movement.day = day;
        movement.moveNumber = moveNumber;
        movement.x = h_agents[i].x;
        movement.y = h_agents[i].y;
        trackedAgents[i].movements.push_back(movement);
    }
}

void Simulation::updateStatistics(int day) {
    // Contar agentes por estado
    int infected = 0;         // Actualmente infectados
    int quarantined = 0;      // En cuarentena
    int recovered = 0;        // Recuperados (nunca se usa directamente)
    int deaths = 0;           // Fallecidos
    int healthy = 0;          // Sanos (nunca infectados)

    // Primero contamos el estado actual de los agentes
    for (int i = 0; i < numAgents; i++) {
        switch (h_agents[i].status) {
        case NOT_INFECTED:
            healthy++;
            break;
        case INFECTED:
            infected++;
            break;
        case QUARANTINED:
            quarantined++;
            break;
        case DECEASED:
            deaths++;
            break;
        }
    }

    // Para el día 0, inicializamos con valores sensatos
    if (day == 0) {
        DailyStats firstDayStats;
        firstDayStats.infected = infected;      // Nuevos infectados este día
        firstDayStats.recovered = 0;            // Nuevos recuperados este día
        firstDayStats.deaths = 0;               // Nuevos fallecidos este día
        firstDayStats.totalInfected = infected; // Total acumulado de infectados
        firstDayStats.totalRecovered = 0;       // Total acumulado de recuperados
        firstDayStats.totalDeaths = 0;          // Total acumulado de fallecidos
        firstDayStats.currentInfected = infected;       // Actualmente infectados
        firstDayStats.currentQuarantined = quarantined; // Actualmente en cuarentena

        stats.push_back(firstDayStats);
    }
    else {
        // Para días posteriores, calculamos incrementos respecto al día anterior
        DailyStats todayStats;

        // Calculamos nuevos casos del día (pueden ser positivos o negativos)
        int newInfected = (infected + quarantined) - (stats[day - 1].currentInfected + stats[day - 1].currentQuarantined);

        // Si newInfected es negativo, significa que hay recuperados o fallecidos
        // Los nuevos infectados no pueden ser negativos en realidad
        todayStats.infected = std::max(0, newInfected);

        // Calculamos recuperados este día 
        // Un agente se recupera si: estaba infectado o en cuarentena ayer, y hoy está sano
        int prevActive = stats[day - 1].currentInfected + stats[day - 1].currentQuarantined;
        int currActive = infected + quarantined;
        int newRecovered = 0;

        // Si tenemos menos infectados activos que ayer y no todos se murieron...
        if (currActive < prevActive) {
            // Calculamos cuántos se recuperaron (en vez de morir)
            int newDeaths = deaths - stats[day - 1].totalDeaths;
            newRecovered = prevActive - currActive - newDeaths;
        }

        todayStats.recovered = std::max(0, newRecovered);

        // Muertes nuevas (esto debería ser correcto)
        todayStats.deaths = deaths - stats[day - 1].totalDeaths;

        // Acumulados
        todayStats.totalInfected = stats[day - 1].totalInfected + todayStats.infected;
        todayStats.totalRecovered = stats[day - 1].totalRecovered + todayStats.recovered;
        todayStats.totalDeaths = deaths; // Acumulado de muertes es el total actual

        // Valores actuales
        todayStats.currentInfected = infected;
        todayStats.currentQuarantined = quarantined;

        stats.push_back(todayStats);
    }

    // Registrar días importantes
    int totalInfected = stats[day].totalInfected;
    int totalRecovered = stats[day].totalRecovered;
    int totalDeaths = stats[day].totalDeaths;

    // Días de infección
    if (totalInfected > 0 && firstInfectionDay == -1) {
        firstInfectionDay = day;
    }
    if (totalInfected >= numAgents / 2 && halfInfectionDay == -1) {
        halfInfectionDay = day;
    }
    if (totalInfected >= numAgents && allInfectionDay == -1) {
        allInfectionDay = day;
    }

    // Días de recuperación
    if (totalRecovered > 0 && firstRecoveryDay == -1) {
        firstRecoveryDay = day;
    }
    if (totalRecovered >= numAgents / 2 && halfRecoveryDay == -1) {
        halfRecoveryDay = day;
    }
    if (totalRecovered >= numAgents && allRecoveryDay == -1) {
        allRecoveryDay = day;
    }

    // Días de muertes
    if (totalDeaths > 0 && firstDeathDay == -1) {
        firstDeathDay = day;
    }
    if (totalDeaths >= numAgents / 2 && halfDeathDay == -1) {
        halfDeathDay = day;
    }
    if (totalDeaths >= numAgents && allDeathDay == -1) {
        allDeathDay = day;
    }

    // Guardar estadísticas del día en un archivo según el tipo de simulación
    std::string suffix = (currentSimType == CPU_SIMULATION) ? "_cpu" : "_gpu";
    saveStatsToFile(suffix);
}

// Método para guardar estadísticas a un archivo con sufijo específico
void Simulation::saveStatsToFile(const std::string& suffix) {
    std::ofstream dailyStatsFile;
    int day = stats.size() - 1; // Obtener el día actual

    if (day == 0) {
        // Crear o sobrescribir el archivo en el primer día
        dailyStatsFile.open("daily_stats" + suffix + ".csv");

        // Escribir encabezados
        dailyStatsFile << "Día,Infectados,En Cuarentena,Total Infectados Activos,Recuperados Hoy,";
        dailyStatsFile << "Total Recuperados,Fallecidos Hoy,Total Fallecidos,Nuevos Contagios\n";
    }
    else {
        // Abrir en modo append para días posteriores
        dailyStatsFile.open("daily_stats" + suffix + ".csv", std::ios::app);
    }

    if (dailyStatsFile.is_open()) {
        // Escribir datos del día
        dailyStatsFile << day << ","
            << stats[day].currentInfected << ","
            << stats[day].currentQuarantined << ","
            << (stats[day].currentInfected + stats[day].currentQuarantined) << ","
            << stats[day].recovered << ","
            << stats[day].totalRecovered << ","
            << stats[day].deaths << ","
            << stats[day].totalDeaths << ","
            << stats[day].infected << "\n";

        dailyStatsFile.close();
    }
}

void Simulation::runCPU() {
    // Establecer tipo de simulación
    currentSimType = CPU_SIMULATION;

    // Limpiar estadísticas previas
    stats.clear();

    // Limpiar seguimientos previos excepto la posición inicial
    for (int i = 0; i < numTrackedAgents; i++) {
        // Mantener solo la posición inicial (elemento 0)
        if (!trackedAgents[i].movements.empty()) {
            AgentMovement initialPos = trackedAgents[i].movements[0];
            trackedAgents[i].movements.clear();
            trackedAgents[i].movements.push_back(initialPos);
        }
    }

    for (int day = 0; day < MAX_DAYS; day++) {
        // Para cada movimiento del día
        for (int move = 0; move < MAX_MOVEMENTS; move++) {
            // Aplicar regla 1 (contagio) y regla 2 (movilidad)
            applyRule1CPU(h_agents, numAgents, CONTAGION_RADIUS);
            applyRule2CPU(h_agents, numAgents, MAX_LOCAL_RADIUS, AREA_WIDTH, AREA_HEIGHT);

            // Solo mantenemos el seguimiento para visualización de algunos agentes
            trackAgentMovements(day, move + 1);
        }

        // Al final del día, aplicar regla 3 (contagio externo)
        applyRule3CPU(h_agents, numAgents);

        // Aplicar regla 4 (incubación, cuarentena y recuperación)
        applyRule4CPU(h_agents, numAgents);

        // Aplicar regla 5 (casos fatales)
        applyRule5CPU(h_agents, numAgents);

        // Actualizar estadísticas
        updateStatistics(day);
    }
}

void Simulation::runGPU() {
    // Establecer tipo de simulación
    currentSimType = GPU_SIMULATION;

    // Inicializar CUDA
    initializeCUDA();

    // Limpiar estadísticas previas
    stats.clear();

    // Limpiar seguimientos previos excepto la posición inicial
    for (int i = 0; i < numTrackedAgents; i++) {
        // Mantener solo la posición inicial (elemento 0)
        if (!trackedAgents[i].movements.empty()) {
            AgentMovement initialPos = trackedAgents[i].movements[0];
            trackedAgents[i].movements.clear();
            trackedAgents[i].movements.push_back(initialPos);
        }
    }

    // Configuración de bloques
    int blocks = (numAgents + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int day = 0; day < MAX_DAYS; day++) {
        // Para cada movimiento del día
        for (int move = 0; move < MAX_MOVEMENTS; move++) {
            // Aplicar regla 1 (contagio)
            applyRule1Kernel << <blocks, BLOCK_SIZE >> > (d_agents, numAgents, d_states, CONTAGION_RADIUS);
            cudaDeviceSynchronize();
            cudaCheckError();

            // Aplicar regla 2 (movilidad)
            applyRule2Kernel << <blocks, BLOCK_SIZE >> > (d_agents, numAgents, d_states, MAX_LOCAL_RADIUS, AREA_WIDTH, AREA_HEIGHT);
            cudaDeviceSynchronize();
            cudaCheckError();

            // Solo copiar para visualización
            if (move % 5 == 0 || move == MAX_MOVEMENTS - 1) {
                cudaMemcpy(h_agents, d_agents, numAgents * sizeof(Agent), cudaMemcpyDeviceToHost);
                cudaCheckError();

                // Rastrear movimientos para visualización
                trackAgentMovements(day, move + 1);
            }
        }

        // Al final del día, aplicar regla 3 (contagio externo)
        applyRule3Kernel << <blocks, BLOCK_SIZE >> > (d_agents, numAgents, d_states);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Aplicar regla 4 (incubación, cuarentena y recuperación)
        applyRule4Kernel << <blocks, BLOCK_SIZE >> > (d_agents, numAgents);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Aplicar regla 5 (casos fatales)
        applyRule5Kernel << <blocks, BLOCK_SIZE >> > (d_agents, numAgents, d_states);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Copiar resultados a CPU para estadísticas
        cudaMemcpy(h_agents, d_agents, numAgents * sizeof(Agent), cudaMemcpyDeviceToHost);
        cudaCheckError();

        // Actualizar estadísticas
        updateStatistics(day);
    }

    // Limpiar recursos CUDA
    cleanupCUDA();
}

void Simulation::printResults() {
    std::cout << "\n===== SIMULATION RESULTS =====\n";

    // Tabla de estadísticas diarias
    std::cout << "\nDaily Statistics:\n";
    std::cout << std::setw(5) << "Day"
        << std::setw(10) << "Infected"
        << std::setw(12) << "Quarantined"
        << std::setw(12) << "NewCases"
        << std::setw(10) << "Recovered"
        << std::setw(10) << "Deaths"
        << std::setw(15) << "TotalInfected"
        << std::setw(15) << "TotalRecovered"
        << std::setw(15) << "TotalDeaths" << std::endl;
    std::cout << std::string(104, '-') << std::endl;

    for (size_t i = 0; i < stats.size(); i++) {
        std::cout << std::setw(5) << i
            << std::setw(10) << stats[i].currentInfected
            << std::setw(12) << stats[i].currentQuarantined
            << std::setw(12) << stats[i].infected
            << std::setw(10) << stats[i].recovered
            << std::setw(10) << stats[i].deaths
            << std::setw(15) << stats[i].totalInfected
            << std::setw(15) << stats[i].totalRecovered
            << std::setw(15) << stats[i].totalDeaths << std::endl;
    }

    // Días clave
    std::cout << "\nKey Milestones:\n";

    // Usar variables auxiliares para manejar los casos de -1
    std::string firstInfectionStr = (firstInfectionDay != -1) ? std::to_string(firstInfectionDay) : "N/A";
    std::string halfInfectionStr = (halfInfectionDay != -1) ? std::to_string(halfInfectionDay) : "N/A";
    std::string allInfectionStr = (allInfectionDay != -1) ? std::to_string(allInfectionDay) : "N/A";

    std::string firstRecoveryStr = (firstRecoveryDay != -1) ? std::to_string(firstRecoveryDay) : "N/A";
    std::string halfRecoveryStr = (halfRecoveryDay != -1) ? std::to_string(halfRecoveryDay) : "N/A";
    std::string allRecoveryStr = (allRecoveryDay != -1) ? std::to_string(allRecoveryDay) : "N/A";

    std::string firstDeathStr = (firstDeathDay != -1) ? std::to_string(firstDeathDay) : "N/A";
    std::string halfDeathStr = (halfDeathDay != -1) ? std::to_string(halfDeathDay) : "N/A";
    std::string allDeathStr = (allDeathDay != -1) ? std::to_string(allDeathDay) : "N/A";

    std::cout << "First Infection: " << firstInfectionStr << " day(s)\n";
    std::cout << "50% Infected: " << halfInfectionStr << " day(s)\n";
    std::cout << "100% Infected: " << allInfectionStr << " day(s)\n";

    std::cout << "First Recovery: " << firstRecoveryStr << " day(s)\n";
    std::cout << "50% Recovered: " << halfRecoveryStr << " day(s)\n";
    std::cout << "100% Recovered: " << allRecoveryStr << " day(s)\n";

    std::cout << "First Death: " << firstDeathStr << " day(s)\n";
    std::cout << "50% Deaths: " << halfDeathStr << " day(s)\n";
    std::cout << "100% Deaths: " << allDeathStr << " day(s)\n";

    // Estadísticas finales
    size_t lastDay = stats.size() - 1;
    std::cout << "\nFinal Statistics:\n";
    std::cout << "Total Infected: " << stats[lastDay].totalInfected << " agents ("
        << (float)stats[lastDay].totalInfected / numAgents * 100.0f << "%)\n";
    std::cout << "Total Recovered: " << stats[lastDay].totalRecovered << " agents ("
        << (float)stats[lastDay].totalRecovered / numAgents * 100.0f << "%)\n";
    std::cout << "Total Deaths: " << stats[lastDay].totalDeaths << " agents ("
        << (float)stats[lastDay].totalDeaths / numAgents * 100.0f << "%)\n";

    // Indicar que los archivos de registro están disponibles
    std::string logType = (currentSimType == CPU_SIMULATION) ? "CPU" : "GPU";
    std::cout << "\nDetailed agent logs have been written to the 'agent_logs_" << logType << "' directory.\n";
    std::cout << "Summary of daily statistics is available in 'daily_stats_" << logType << ".csv'.\n";
}

void Simulation::printAgentTracking() {
    std::cout << "\n===== AGENT TRACKING =====\n";

    // Solo imprimir algunos días y movimientos para no saturar la salida
    int daysToShow = 2; // Solo mostrar los primeros 2 días

    for (int i = 0; i < numTrackedAgents; i++) {
        std::cout << "\nAgente " << trackedAgents[i].id << ":\n";
        std::cout << "Atributos:\n";
        std::cout << "- p_con (Prob. Contagio): " << trackedAgents[i].p_con << "\n";
        std::cout << "- p_ext (Prob. Contagio Externo): " << trackedAgents[i].p_ext << "\n";
        std::cout << "- p_fat (Prob. Mortalidad): " << trackedAgents[i].p_fat << "\n";
        std::cout << "- p_mov (Prob. Movilidad): " << trackedAgents[i].p_mov << "\n";
        std::cout << "- p_smo (Prob. Movilidad Local): " << trackedAgents[i].p_smo << "\n";

        // Mostrar posición inicial
        if (!trackedAgents[i].movements.empty()) {
            std::cout << "Posición Inicio (x,y): ("
                << trackedAgents[i].movements[0].x << ", "
                << trackedAgents[i].movements[0].y << ")\n";
        }

        // Rastrear movimientos por día
        int currentDay = -1;

        for (const auto& move : trackedAgents[i].movements) {
            // Solo mostrar algunos días para que la salida no sea muy larga
            if (move.day >= daysToShow) continue;

            // Si cambiamos de día, mostrar encabezado
            if (move.day != currentDay) {
                currentDay = move.day;
                std::cout << "Día " << currentDay << ":\n";
            }

            // Mostrar movimiento si no es la posición inicial
            if (move.moveNumber > 0) {
                std::cout << "  Movimiento " << move.moveNumber << " (x,y): ("
                    << move.x << ", " << move.y << ")\n";
            }
        }
    }
}