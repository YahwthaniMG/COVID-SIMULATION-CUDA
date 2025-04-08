# Simulador de Propagación COVID-19 con CUDA

Este proyecto implementa un modelo basado en agentes para simular la propagación del COVID-19 en un espacio de trabajo, utilizando C++ con aceleración mediante CUDA para comparar el rendimiento entre CPU y GPU.

## Descripción

El simulador crea un entorno de 500m x 500m donde 1024 agentes se mueven e interactúan, siguiendo un conjunto de reglas que modelan la transmisión del virus, la movilidad, la cuarentena, la recuperación y los casos fatales. El objetivo es analizar cómo diferentes factores afectan la propagación del COVID-19 y comparar el rendimiento de la implementación en CPU frente a GPU.

## Características Principales

- **Simulación basada en agentes**: 1024 agentes con características individualizadas (probabilidades de contagio, movilidad, mortalidad, etc.)
- **Implementación en CPU y GPU**: Comparación directa de rendimiento entre procesadores secuenciales y paralelos
- **Modelo de 5 reglas**: 
  1. Contagio entre agentes cercanos
  2. Patrones de movilidad (movimientos cortos y largos)
  3. Contagio externo (infección fuera del espacio simulado)
  4. Incubación, cuarentena y recuperación
  5. Modelado de casos fatales
- **Análisis detallado**: Estadísticas diarias y acumuladas de la propagación del virus
- **Visualización de resultados**: Datos de salida en formato CSV para análisis posterior

## Implementación Técnica

### Parámetros de la Simulación

- **Espacio**: Área de 500m x 500m
- **Población**: 1024 agentes
- **Duración**: 30 días simulados
- **Movimientos diarios**: 10 movimientos por día por agente
- **Distancia de contagio**: 1 metro

### Atributos de los Agentes

Cada agente posee atributos individuales que determinan su comportamiento:

- Probabilidad de contagio (0.02-0.03)
- Probabilidad de contagio externo (0.02-0.03)
- Probabilidad de mortalidad (0.007-0.07)
- Probabilidad de movilidad (0.3-0.5)
- Probabilidad de movilidad de corta distancia (0.7-0.9)
- Tiempo de incubación (5-6 días)
- Tiempo de recuperación (14 días)

### Estados Posibles de los Agentes

- **No infectado (0)**: Agente sano susceptible a infección
- **Infectado (1)**: Contagiado pero sin síntomas aparentes (en incubación)
- **En cuarentena (-1)**: Contagiado con síntomas visibles, en aislamiento
- **Fallecido (-2)**: Agente que no sobrevivió a la infección

## Análisis de Rendimiento

El proyecto ejecuta la misma simulación tanto en CPU como en GPU, midiendo y comparando los tiempos de ejecución. Esto proporciona una demostración práctica de las ventajas de la computación paralela en CUDA para simulaciones complejas.

## Estadísticas Generadas

La simulación produce los siguientes datos:

- Número de casos acumulados de agentes contagiados
- Nuevos casos positivos por día
- Casos de recuperación acumulados y diarios
- Casos fatales acumulados y diarios
- Días clave en la progresión de la epidemia (primer contagio, 50% de población infectada, etc.)

## Requisitos del Sistema

- CUDA Toolkit 12.0 o superior
- Visual Studio 2022 con soporte para C++17
- GPU compatible con CUDA

## Uso

1. Compilar el proyecto en Visual Studio
2. Ejecutar el archivo binario generado
3. La aplicación automáticamente ejecutará la simulación en CPU y luego en GPU
4. Al finalizar, se generarán archivos CSV con estadísticas y se mostrará el factor de aceleración obtenido

## Archivos de Salida

- **daily_stats_cpu.csv**: Estadísticas diarias de la simulación en CPU
- **daily_stats_gpu.csv**: Estadísticas diarias de la simulación en GPU

## Estructura del Proyecto

- **agent.h**: Definición de la estructura y comportamiento de los agentes
- **config.h**: Configuración de parámetros de la simulación
- **rules.h/rules.cu**: Implementación de las 5 reglas del modelo
- **simulation.h/simulation.cu**: Clase principal que gestiona la simulación
- **main.cu**: Punto de entrada del programa
- **utils.h/utils.cu**: Funciones auxiliares

## Autores

- Yahwthani Morales Gómez
- Mario Alejandro Rodriguez Gonzalez

## Licencia

Este proyecto es parte de una evaluación académica para la Universidad Panamericana.