"""
main.py
API de inferencia con FastAPI. Sirve el modelo con el alias "champion" de MLflow.

Ejecución (PowerShell, desde la raíz del proyecto):
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    $env:PYTHONUTF8 = "1"
    poetry run uvicorn api.main:app --port 8000

Documentación interactiva: http://localhost:8000/docs

El modelo se carga UNA vez, al arrancar la API. Si el entrenamiento asigna un
champion nuevo, hay que reiniciar la API (Ctrl + C y volver a ejecutar el comando).
"""

import os
import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from src.load_data import leer_config  # noqa: E402

# ------------------------------------------------ Cargar el champion al iniciar
cfg = leer_config()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))

modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
VERSION = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version
print(f"API lista con {NOMBRE_MODELO} versión {VERSION}")

app = FastAPI(
    title="API de salario alto en vacantes de IA y datos",
    description=(
        "Predice si una vacante de IA o ciencia de datos ofrece salario alto "
        f"(salary >= {cfg['objetivo']['umbral_salario']}) con el modelo champion de MLflow."
    ),
)


class Vacante(BaseModel):
    """Datos de una vacante. Los nombres son iguales a las columnas del CSV (sin job_id ni salary)."""

    job_title: str
    company_size: str
    company_industry: str
    country: str
    remote_type: str
    experience_level: str
    years_experience: int
    education_level: str
    skills_python: int
    skills_sql: int
    skills_ml: int
    skills_deep_learning: int
    skills_cloud: int
    job_posting_month: int
    job_posting_year: int
    hiring_urgency: str
    job_openings: int

    # Ejemplo que aparece ya escrito en http://localhost:8000/docs
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
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
                    "job_openings": 3,
                }
            ]
        }
    )


@app.get("/health")
def health():
    """Comprueba que la API está viva y qué versión del modelo usa."""
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}


@app.post("/predict")
def predict(vacante: Vacante):
    """Predice si la vacante tiene salario alto."""
    datos = pd.DataFrame([vacante.model_dump()])
    # El modelo usa solo las columnas con las que se entrenó.
    datos = datos[list(modelo.feature_names_in_)]
    probabilidad = float(modelo.predict_proba(datos)[0, 1])
    return {
        "probabilidad_salario_alto": round(probabilidad, 4),
        "prediccion": int(probabilidad >= UMBRAL),
        "interpretacion": "salario alto" if probabilidad >= UMBRAL else "salario no alto",
        "version_modelo": VERSION,
    }
