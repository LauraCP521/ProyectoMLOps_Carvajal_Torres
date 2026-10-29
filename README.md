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