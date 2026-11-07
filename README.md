# ProyectoMLOps_Carvajal_Torres

## Integrantes:
### Angela Viviana Torres Velandia
### Laura Carolina Carvajal Parra

## Caso de uso y dataset

Caso de uso: mercado laboral de IA y ciencia de datos (2020–2026).

- **Nombre:** AI & Data Science Job Market Dataset (2020–2026)
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/shree0910/ai-and-data-science-job-market-dataset-20202026)
- **Archivo:** `data/AI_Job_Market_Dataset.csv` (aprox. 1 MB, se versiona en Git)
- **Tamaño:** 10 345 filas y 19 columnas, sin valores vacíos.
- **Columnas:** `job_id`, `job_title`, `company_size`, `company_industry`, `country`, `remote_type`, `experience_level`, `years_experience`, `education_level`, `skills_python`, `skills_sql`, `skills_ml`, `skills_deep_learning`, `skills_cloud`, `salary`, `job_posting_month`, `job_posting_year`, `hiring_urgency`, `job_openings`.

Si el archivo no está en `data/`, descárgalo desde el enlace de Kaggle (botón **Download**), descomprímelo y guarda el CSV en la carpeta `data/` con el nombre `AI_Job_Market_Dataset.csv`.

La ruta del dataset está en `configs/config.yaml`.

## Comprobar los datos

Desde la raíz del proyecto, ejecuta: `poetry run python src/load_data.py`

Debe mostrar 10 345 filas y 19 columnas.

## Requisitos
- Python 3.12 a 3.14
- Poetry 2.5.1
- Git
- Docker Desktop

## Instalación
```
git clone https://github.com/LauraCP521/ProyectoMLOps_Carvajal_Torres.git
cd ProyectoMLOps_Carvajal_Torres
poetry install
```

## Levantar servicios
Desde la carpeta services/ del repositorio del curso:
```
docker compose up -d
```

## Ejecutar el flow de prueba
```
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/prueba_flow.py
```

## Actividad 2: pipelines de entrenamiento e inferencia, y API

### Problema que resuelve el modelo

Predecir si una vacante de IA o ciencia de datos ofrece **salario alto**: `salario_alto = 1` si `salary >= 113000` (aproximadamente la mediana del dataset) y `0` si no. Es una clasificación binaria, evaluada con AUC.

La variable `salary` nunca se usa como entrada del modelo, porque de ella sale la respuesta. Tampoco se usa `job_id`.

### Estructura

```
ProyectoMLOps_Carvajal_Torres/
├── api/
│   └── main.py                  # API de FastAPI que sirve el modelo champion
├── configs/
│   ├── config.yaml              # datos, objetivo, modelo, MLflow, registro, inferencia y horario
│   └── features.yaml            # selección de variables: ALL, NUMERIC o CUSTOM
├── data/
│   ├── AI_Job_Market_Dataset.csv
│   └── predicciones/            # CSV diarios de predicciones (no se suben a Git)
├── flows/
│   ├── entrenamiento_flow.py    # ingesta -> entrenamiento -> evaluación -> registro
│   ├── inferencia_flow.py       # usa el modelo champion y guarda predicciones con fecha
│   ├── desplegar.py             # despliegues en Prefect (inferencia diaria)
│   └── prueba_flow.py           # flow de validación de la actividad 1
├── src/
│   ├── load_data.py             # carga y validación del CSV
│   └── modelo.py                # lógica del modelo (sin Prefect ni MLflow)
└── tests/
    └── test_modelo.py           # pruebas de la lógica del modelo
```

### Variables de entorno

En **cada** terminal donde se ejecute un flow o la API:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

### 1. Entrenamiento

```powershell
poetry run python flows/entrenamiento_flow.py
```

- Cada ejecución crea un run en MLflow (experimento `mercado-laboral-ia`) con el tag `fase = entrenamiento`, los parámetros `features` y `n_features`, y las métricas `auc`, `accuracy`, `precision`, `recall` y `f1`.
- El modelo pasa a ser `champion` (modelo `modelo-salario-alto` en el Model Registry) solo si su AUC es mayor o igual que `registro.umbral_auc` (0.65) **y** supera el AUC del champion actual en al menos `registro.mejora_minima` (0.001). Si no, el run queda guardado y el alias no cambia.
- El tag `resultado_registro` del run indica qué pasó: `nuevo_champion`, `no_supera_champion` o `bajo_minimo`.

### 2. Inferencia

```powershell
poetry run python flows/inferencia_flow.py
```

Carga `models:/modelo-salario-alto@champion`, toma 100 vacantes al azar (simulan vacantes nuevas) y guarda `data/predicciones/prediccion_<fecha>.csv`. Crea un run en MLflow con el tag `fase = inferencia` y la versión del modelo usada.

### 3. Despliegues en Prefect

```powershell
poetry run python flows/desplegar.py
```

- `entrenamiento-salario/entrenamiento-manual`: sin horario, se lanza a mano.
- `inferencia-salario/inferencia-diaria`: todos los días a las 06:00 a. m., hora de Colombia.

La terminal debe quedar abierta para que Prefect pueda lanzar las ejecuciones.

### 4. API

```powershell
poetry run uvicorn api.main:app --port 8000
```

- Documentación interactiva: http://localhost:8000/docs
- Estado y versión del modelo: http://localhost:8000/health
- Predicción: `POST http://localhost:8000/predict`

Si cambia el champion, hay que reiniciar la API.

### Predicción de ejemplo

Petición:

```json
{
  "job_title": "Machine Learning Engineer",
  "company_size": "MNC",
  "company_industry": "Technology",
  "country": "USA",
  "remote_type": "Remote",
  "experience_level": "Senior",
  "years_experience": 9,
  "education_level": "Master",
  "skills_python": 1,
  "skills_sql": 1,
  "skills_ml": 1,
  "skills_deep_learning": 1,
  "skills_cloud": 1,
  "job_posting_month": 6,
  "job_posting_year": 2026,
  "hiring_urgency": "High",
  "job_openings": 3
}
```

Respuesta:

```json
{
  "probabilidad_salario_alto": 1.0,
  "prediccion": 1,
  "interpretacion": "salario alto",
  "version_modelo": "2"
}
```

### Pruebas

```powershell
poetry run pytest
```
