"""
Tests del MODELO (no necesitan MLflow ni Prefect).

Ejecución (desde la raíz del proyecto):
    poetry run pytest tests/test_modelo.py
"""

import numpy as np
import pandas as pd
import pytest

from src.load_data import leer_config, load_data
from src.modelo import (
    NUMERICAS,
    VARIABLES,
    calcular_metricas,
    construir_modelo,
    crear_objetivo,
    dividir,
    importancia_variables,
    resolver_features,
)


def test_objetivo_salario_alto():
    df = pd.DataFrame({"salary": [100000, 113000, 150000]})
    cfg = {"columna_origen": "salary", "umbral_salario": 113000}
    assert crear_objetivo(df, cfg).tolist() == [0, 1, 1]


def test_seleccion_all_y_numeric():
    _, num, cat = resolver_features({"seleccion": "ALL"})
    assert len(num + cat) == len(VARIABLES)

    _, num, cat = resolver_features({"seleccion": "NUMERIC"})
    assert num == NUMERICAS and cat == []


def test_seleccion_custom_rechaza_columnas_que_no_existen():
    with pytest.raises(ValueError):
        resolver_features({"seleccion": "CUSTOM", "custom": ["columna_falsa"]})


def test_salary_nunca_es_variable():
    assert "salary" not in VARIABLES
    assert "job_id" not in VARIABLES


def test_dividir_es_reproducible_y_sin_solapamiento():
    cfg = leer_config()
    df = load_data()
    ent1, pru1 = dividir(df, cfg["objetivo"], cfg["division"])
    ent2, pru2 = dividir(df, cfg["objetivo"], cfg["division"])
    assert ent1.index.equals(ent2.index) and pru1.index.equals(pru2.index)
    assert len(set(ent1.index) & set(pru1.index)) == 0
    assert len(pru1) == round(len(df) * cfg["division"]["test_size"])
    # La estratificación mantiene la misma tasa del objetivo en los dos grupos.
    nombre = cfg["objetivo"]["nombre"]
    assert ent1[nombre].mean() == pytest.approx(pru1[nombre].mean(), abs=0.01)


@pytest.mark.parametrize("tipo, parametros", [
    ("logistic_regression", {"C": 1.0, "max_iter": 1000}),
    ("random_forest", {"n_estimators": 10, "max_depth": 3, "random_state": 42}),
])
def test_modelo_predice_probabilidades_validas(tipo, parametros):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "years_experience": rng.integers(0, 15, 200),
        "skills_python": rng.integers(0, 2, 200),
        "experience_level": rng.choice(["Entry", "Mid", "Senior"], 200),
    })
    y = (X["experience_level"] == "Senior").astype(int)
    modelo = construir_modelo(tipo, parametros, ["years_experience", "skills_python"], ["experience_level"])
    modelo.fit(X, y)
    proba = modelo.predict_proba(X)[:, 1]
    assert proba.shape == (200,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calcular_metricas_perfecto():
    metricas = calcular_metricas(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), umbral=0.5)
    assert metricas["auc"] == 1.0 and metricas["accuracy"] == 1.0


def test_importancia_detecta_la_variable_que_decide():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "experience_level": rng.choice(["Entry", "Mid", "Senior"], 300),
        "job_posting_month": rng.integers(1, 13, 300),
    })
    y = (X["experience_level"] == "Senior").astype(int)
    modelo = construir_modelo("logistic_regression", {"max_iter": 1000}, ["job_posting_month"], ["experience_level"])
    modelo.fit(X, y)
    tabla = importancia_variables(modelo, X, y)
    assert tabla.iloc[0]["variable"] == "experience_level"
