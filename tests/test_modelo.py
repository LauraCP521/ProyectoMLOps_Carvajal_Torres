"""
Pruebas de la lógica del modelo (no necesitan MLflow ni Prefect).

Ejecución (desde la raíz del proyecto):
    poetry run pytest
"""

import pandas as pd
import pytest

from src.modelo import NUMERICAS, VARIABLES, crear_objetivo, resolver_features


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
