# ProyectoMLOps_Carvajal_Torres

## Integrantes:
### Angela Viviana Torres Velandia
### Laura Carolina Carvajal Parra

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