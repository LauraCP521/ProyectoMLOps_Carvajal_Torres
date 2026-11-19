"""
Tests de la API con un modelo FALSO (mock): no necesitan MLflow encendido.

Se reemplazan (monkeypatch) las funciones de MLflow que usa api/main.py al arrancar,
y se prueba la API con TestClient de FastAPI.

Ejecución (desde la raíz del proyecto):
    poetry run pytest tests/test_api.py
"""

import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.modelo import VARIABLES

VACANTE = {
    "job_title": "Machine Learning Engineer", "company_size": "MNC", "company_industry": "Technology",
    "country": "USA", "remote_type": "Remote", "experience_level": "Senior", "years_experience": 9,
    "education_level": "Master", "skills_python": 1, "skills_sql": 1, "skills_ml": 1,
    "skills_deep_learning": 1, "skills_cloud": 1, "job_posting_month": 6, "job_posting_year": 2026,
    "hiring_urgency": "High", "job_openings": 3,
}


class ModeloFalso:
    feature_names_in_ = np.array(VARIABLES)

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    # Se reemplaza MLflow ANTES de importar la API, porque main.py carga el modelo al importarse.
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")))
    sys.modules.pop("api.main", None)

    import api.main as main

    return TestClient(main.app)


def test_health(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["version"] == "7"


def test_predict_devuelve_prediccion(client):
    respuesta = client.post("/predict", json=VACANTE)
    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "probabilidad_salario_alto": 0.7,
        "prediccion": 1,
        "interpretacion": "salario alto",
        "version_modelo": "7",
    }


def test_predict_rechaza_vacante_incompleta(client):
    incompleta = {k: v for k, v in VACANTE.items() if k != "country"}
    assert client.post("/predict", json=incompleta).status_code == 422
