# proyecto_tripulaciones - The Bridge Madrid - Sep 2024

# Proyecto de Ingeniería de Datos - ETL, Predicción ML y Chatbot

## Descripción

Este repositorio fue creado como parte de un proyecto colaborativo en el marco de un bootcamp de análisis de datos. Su propósito es centralizar el trabajo realizado por el equipo de ingeniería y análisis de datos, incluyendo:

- El diseño e implementación de un flujo ETL para recolección y estructuración de datos.
- La creación de un modelo de machine learning para tareas de predicción.
- El desarrollo e integración de un chatbot en la aplicación final.

Este repositorio también sirve como control de versiones del código y punto central de colaboración entre los distintos equipos.

---

## Componentes del Proyecto

### 🔄 Proceso ETL

#### 1. Identificación de Fuentes
- Recolección de datos desde APIs y sitios web.
- Evaluación de confiabilidad, frecuencia y estructura.

#### 2. Verificación Legal
- Validación de términos de uso y licencias.

#### 3. Extracción
- Automatización mediante scraping e integraciones con APIs.

#### 4. Transformación
- Limpieza y estandarización con `pandas`.

#### 5. Carga
- Almacenamiento en PostgreSQL en la nube (Render).

---

### 🤖 Módulo de Predicción con Machine Learning

Ubicado en el directorio `0_ml_prediction/`, este módulo contiene:

- Scripts de entrenamiento de modelos.
- Evaluaciones de rendimiento.
- Subentorno virtual para aislar dependencias específicas.

#### Tecnologías utilizadas
- Scikit-learn
- Pandas / Numpy
- Jupyter Notebooks

### 💬 Chatbot Integrado
Como parte de la funcionalidad de la aplicación, se integró un chatbot que interactúa con los usuarios finales. Este chatbot se alimenta de los guardados en la base de datos que almacena los datos que alimentan la aplicación.

Características:
Interacción natural e amigable con usuarios.

Acceso a datos estructurados procesados por el pipeline ETL.

Posibilidad de ampliar funcionalidades con nuevas intenciones/respuestas.

Tecnologías utilizadas:
Procesamiento de lenguaje natural (NLP)

Backend conectado a base de datos

### Estructura del Repositorio

├── 0_ml_prediction/       # Subentorno virtual y scripts de ML
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   ├── visuals/
│   └── requirements.txt
├── data/                  # Datasets originales y procesados
    ├── processed/
│   └── raw/
├── my_scripts/            # Scripts de ETL
├── notebooks/             # Notebooks de análisis y pruebas
└── README.md              
