"""
Tests de la INFERENCIA con un modelo falso (sin MLflow ni Prefect).

Ejecución (desde la raíz del proyecto):
    poetry run pytest tests/test_inferencia.py
"""

import numpy as np
import pytest

from src.inferencia import armar_lote, nombre_archivo_lote, predecir_lote
from src.load_data import leer_config, load_data
from src.modelo import VARIABLES, dividir


class ModeloFalso:
    """Se comporta como el modelo de MLflow: tiene feature_names_in_ y predict_proba."""

    feature_names_in_ = np.array(VARIABLES)

    def predict_proba(self, X):
        # Probabilidad 0.8 para Senior y 0.2 para los demás.
        p = np.where(X["experience_level"] == "Senior", 0.8, 0.2)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def prueba():
    cfg = leer_config()
    _, prueba = dividir(load_data(), cfg["objetivo"], cfg["division"])
    return prueba


def test_lote_sale_de_la_parte_de_prueba(prueba):
    lote = armar_lote(prueba, 500, 0.0, "salario_alto", 42)
    assert len(lote) == 500
    assert set(lote.index) <= set(prueba.index)
    assert (lote["nivel_drift"] == 0.0).all()


def test_lote_trae_etiquetas_reales(prueba):
    lote = armar_lote(prueba, 100, 0.6, "salario_alto", 42)
    assert "salario_alto" in lote.columns and lote["salario_alto"].notna().all()


def test_lote_mas_grande_que_la_prueba_falla(prueba):
    with pytest.raises(ValueError):
        armar_lote(prueba, len(prueba) + 1, 0.0, "salario_alto", 42)


def test_nombre_del_lote_incluye_nivel():
    assert nombre_archivo_lote(0.6, "20261118-100000") == "lote_20261118-100000_nivel_0.6.csv"


def test_predecir_lote_agrega_probabilidad_y_prediccion(prueba):
    lote = armar_lote(prueba, 50, 0.0, "salario_alto", 42)
    resultado = predecir_lote(ModeloFalso(), lote, umbral=0.5)
    assert {"probabilidad_salario_alto", "prediccion"} <= set(resultado.columns)
    assert resultado["probabilidad_salario_alto"].between(0, 1).all()
    senior = resultado["experience_level"] == "Senior"
    assert (resultado.loc[senior, "prediccion"] == 1).all()
    assert (resultado.loc[~senior, "prediccion"] == 0).all()


def test_predecir_lote_falla_si_falta_una_columna(prueba):
    lote = armar_lote(prueba, 10, 0.0, "salario_alto", 42).drop(columns=["country"])
    with pytest.raises(ValueError, match="country"):
        predecir_lote(ModeloFalso(), lote, umbral=0.5)
