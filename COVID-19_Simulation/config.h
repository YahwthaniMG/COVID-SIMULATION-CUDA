#pragma once

// Par�metros de simulaci�n
constexpr int N = 1024;                    // N�mero de agentes
constexpr int MAX_DAYS = 30;               // D�as m�ximos de simulaci�n
constexpr int MAX_MOVEMENTS = 10;          // Movimientos m�ximos por d�a
constexpr float MAX_LOCAL_RADIUS = 5.0f;   // Radio m�ximo para movimientos locales
constexpr float CONTAGION_RADIUS = 1.0f;   // Distancia de contagio
constexpr float AREA_WIDTH = 500.0f;       // Ancho del �rea (metros)
constexpr float AREA_HEIGHT = 500.0f;      // Alto del �rea (metros)

// Rangos de valores para atributos de agentes
constexpr float MIN_CONTAGION_PROB = 0.02f;
constexpr float MAX_CONTAGION_PROB = 0.03f;
constexpr float MIN_EXTERNAL_PROB = 0.02f;
constexpr float MAX_EXTERNAL_PROB = 0.03f;
constexpr float MIN_MORTALITY_PROB = 0.007f;
constexpr float MAX_MORTALITY_PROB = 0.07f;
constexpr float MIN_MOBILITY_PROB = 0.3f;
constexpr float MAX_MOBILITY_PROB = 0.5f;
constexpr float MIN_SHORT_MOBILITY_PROB = 0.7f;
constexpr float MAX_SHORT_MOBILITY_PROB = 0.9f;
constexpr int MIN_INCUBATION_TIME = 5;
constexpr int MAX_INCUBATION_TIME = 6;
constexpr int RECOVERY_TIME = 14;

// Configuraci�n CUDA
constexpr int BLOCK_SIZE = 256;