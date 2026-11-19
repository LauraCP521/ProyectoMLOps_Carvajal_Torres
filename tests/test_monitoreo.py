"""
Tests de la lógica de DRIFT (no necesitan MLflow, Prefect ni Evidently levantado).

Ejecución (desde la raíz del proyecto):
    poetry run pytest tests/test_monitoreo.py
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoreo import (
    COLUMNAS_DRIFT,
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    columnas_afectadas,
    drift_de_target,
    fraccion_desde_evidently,
    hay_alerta,
    validar_lote,
)

OBJETIVO = "salario_alto"


@pytest.fixture
def lote():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "years_experience": rng.integers(0, 15, n),
        "skills_python": rng.integers(0, 2, n),
        "skills_ml": rng.integers(0, 2, n),
        "skills_cloud": rng.integers(0, 2, n),
        "job_openings": rng.integers(1, 10, n),
        "skills_sql": rng.integers(0, 2, n),
        "skills_deep_learning": rng.integers(0, 2, n),
        "job_posting_month": rng.integers(1, 13, n),
        "job_posting_year": rng.integers(2020, 2027, n),
    })
    df[OBJETIVO] = rng.integers(0, 2, n)
    return df


# ------------------------------------------------------------- emulación
def test_nivel_cero_no_cambia_nada(lote):
    assert aplicar_drift(lote, 0.0, OBJETIVO).equals(lote)


@pytest.mark.parametrize("nivel, esperadas", [(0.3, 3), (0.6, 5), (1.0, 9)])
def test_cuantas_variables_cambian_segun_el_nivel(lote, nivel, esperadas):
    con_drift = aplicar_drift(lote, nivel, OBJETIVO)
    cambiadas = [c for c in COLUMNAS_DRIFT if not con_drift[c].equals(lote[c])]
    assert len(cambiadas) == esperadas
    assert cambiadas == columnas_afectadas(nivel)


def test_tasa_del_objetivo_sube_en_proporcion_al_nivel(lote):
    tasa = lote[OBJETIVO].mean()
    con_drift = aplicar_drift(lote, 0.6, OBJETIVO)
    assert con_drift[OBJETIVO].mean() == pytest.approx(tasa + 0.6 * SUBIDA_TASA_MAXIMA, abs=0.01)


def test_binarias_siguen_siendo_binarias(lote):
    con_drift = aplicar_drift(lote, 1.0, OBJETIVO)
    assert set(con_drift["skills_python"].unique()) <= {0, 1}
    assert con_drift["skills_python"].mean() > lote["skills_python"].mean()


def test_nivel_fuera_de_rango_falla(lote):
    with pytest.raises(ValueError):
        aplicar_drift(lote, 1.5, OBJETIVO)


def test_no_modifica_el_lote_original(lote):
    copia = lote.copy()
    aplicar_drift(lote, 0.6, OBJETIVO)
    assert lote.equals(copia)


# ------------------------------------------------------- drift de features
def test_fraccion_desde_evidently():
    resultado = {"metrics": [
        {"metric_name": "ValueDrift(column=salary)", "value": 0.5},
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 5, "share": 0.294}},
    ]}
    assert fraccion_desde_evidently(resultado) == 0.294


def test_fraccion_desde_evidently_sin_metrica_falla():
    with pytest.raises(ValueError):
        fraccion_desde_evidently({"metrics": []})


def test_alerta_solo_si_supera_el_umbral():
    assert hay_alerta(0.29, 0.25) is True
    assert hay_alerta(0.06, 0.25) is False
    assert hay_alerta(0.25, 0.25) is False


# --------------------------------------------------------- drift de evento
def test_validar_lote_rechaza_lote_sin_etiquetas(lote):
    with pytest.raises(ValueError, match="etiquetas"):
        validar_lote(lote.drop(columns=[OBJETIVO]), OBJETIVO)


def test_validar_lote_rechaza_etiquetas_vacias(lote):
    lote = lote.astype({OBJETIVO: "float"})
    lote.loc[0, OBJETIVO] = np.nan
    with pytest.raises(ValueError, match="vacíos"):
        validar_lote(lote, OBJETIVO)


def test_drift_de_target_detecta_cambio_de_tasa():
    referencia = pd.Series([0] * 500 + [1] * 500)
    lote = pd.Series([0] * 160 + [1] * 340)  # 68 %, como el nivel 0.6
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is True


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([0] * 500 + [1] * 500)
    lote = pd.Series([0] * 250 + [1] * 250)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False
