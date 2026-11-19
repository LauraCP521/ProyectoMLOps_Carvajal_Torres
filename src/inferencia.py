"""
inferencia.py
Lógica de la inferencia: armar el lote y calcular predicciones.

NO usa Prefect ni MLflow: el modelo llega como parámetro, así que los tests usan un modelo falso
(tests/test_inferencia.py).
"""

import pandas as pd

from src.monitoreo import aplicar_drift


def armar_lote(prueba: pd.DataFrame, tamano: int, nivel: float, objetivo: str, semilla: int) -> pd.DataFrame:
    """
    Toma `tamano` vacantes de la parte de PRUEBA (que el modelo no vio al entrenar),
    les aplica el drift emulado y anota el nivel usado en la columna `nivel_drift`.
    """
    if tamano > len(prueba):
        raise ValueError(f"El lote pide {tamano} filas, pero la parte de prueba solo tiene {len(prueba)}")
    base = prueba.sample(n=tamano, random_state=semilla)
    lote = aplicar_drift(base, nivel, objetivo, semilla)
    lote["nivel_drift"] = nivel
    return lote


def nombre_archivo_lote(nivel: float, momento: str) -> str:
    """Nombre de la copia del lote: incluye fecha-hora y nivel, para no sobrescribir lotes anteriores."""
    return f"lote_{momento}_nivel_{nivel:.1f}.csv"


def predecir_lote(modelo, lote: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Agrega al lote la probabilidad de salario alto y la predicción (0 o 1)."""
    # El modelo usa las mismas columnas con las que se entrenó.
    columnas = list(modelo.feature_names_in_)
    faltantes = [c for c in columnas if c not in lote.columns]
    if faltantes:
        raise ValueError(f"Al lote le faltan columnas que el modelo necesita: {faltantes}")

    probabilidad = modelo.predict_proba(lote[columnas])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_salario_alto"] = probabilidad.round(4)
    resultado["prediccion"] = (probabilidad >= umbral).astype(int)
    return resultado
