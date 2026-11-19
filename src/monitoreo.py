"""
monitoreo.py
Lógica de drift: emulación del drift en los lotes, drift de features y drift de evento (target).

Este archivo NO usa Prefect ni MLflow, así que se prueba sin servicios (tests/test_monitoreo.py).

CÓMO SE EMULA EL DRIFT (aplicar_drift)
--------------------------------------
El nivel va de 0.0 (sin drift) a 1.0 (drift máximo) y se define en configs/config.yaml
(inferencia.nivel_drift). Con nivel n:

1. Drift de features: se alteran las primeras round(n x 9) variables de COLUMNAS_DRIFT, en ese orden.
   - Variables binarias (skills_*): la mitad de las vacantes que NO pedían la habilidad pasan a pedirla
     (el mercado empieza a exigir más habilidades).
   - Variables numéricas (años de experiencia, vacantes abiertas, mes, año): se suman 2 desviaciones
     estándar y se redondea a entero (se piden más años de experiencia, hay más vacantes...).
   Nivel 0.3 -> 3 variables: years_experience, skills_python, skills_ml
   Nivel 0.6 -> 5 variables: las anteriores + skills_cloud, job_openings

2. Drift de evento (target): la tasa de salario_alto sube n x 0.30 puntos (0.3 -> +9 puntos,
   0.6 -> +18 puntos). Se convierten al azar vacantes con 0 en 1. El lote trae así sus "etiquetas
   reales", que en producción llegarían con retraso.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Orden en que el drift va afectando variables a medida que sube el nivel.
COLUMNAS_DRIFT = [
    "years_experience",
    "skills_python",
    "skills_ml",
    "skills_cloud",
    "job_openings",
    "skills_sql",
    "skills_deep_learning",
    "job_posting_month",
    "job_posting_year",
]

# Subida máxima de la tasa del objetivo (0.30 = +30 puntos con nivel 1.0).
SUBIDA_TASA_MAXIMA = 0.30

# Fracción de "no exige la habilidad" que pasa a "sí la exige" en una variable binaria con drift.
FRACCION_BINARIA = 0.5


def columnas_afectadas(nivel: float) -> list[str]:
    """Variables que se alteran con un nivel de drift dado."""
    return COLUMNAS_DRIFT[: round(nivel * len(COLUMNAS_DRIFT))]


def aplicar_drift(df: pd.DataFrame, nivel: float, objetivo: str, semilla: int = 42) -> pd.DataFrame:
    """Devuelve una copia del lote con drift emulado en features y en el objetivo."""
    if not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, no {nivel}")

    lote = df.copy()
    rng = np.random.default_rng(semilla)

    # 1. Drift de features
    for columna in columnas_afectadas(nivel):
        if set(lote[columna].unique()) <= {0, 1}:
            ceros = lote.index[lote[columna] == 0]
            cuantos = round(FRACCION_BINARIA * len(ceros))
            if cuantos > 0:
                lote.loc[rng.choice(ceros, size=cuantos, replace=False), columna] = 1
        else:
            desviacion = lote[columna].std()
            if pd.notna(desviacion) and desviacion > 0:
                lote[columna] = (lote[columna] + 2 * desviacion).round().astype(int)

    # 2. Drift de evento (target)
    lote[objetivo] = _subir_tasa(lote[objetivo], nivel * SUBIDA_TASA_MAXIMA, rng)
    return lote


def _subir_tasa(objetivo: pd.Series, subida: float, rng) -> pd.Series:
    """Convierte ceros en unos (al azar) hasta que la tasa de unos sube `subida` puntos."""
    valores = objetivo.astype(int)
    meta = round(min(valores.mean() + subida, 1.0) * len(valores))
    faltan = meta - int(valores.sum())
    if faltan <= 0:
        return objetivo

    ceros = valores.index[valores == 0]
    faltan = min(faltan, len(ceros))
    resultado = valores.copy()
    resultado.loc[rng.choice(ceros, size=faltan, replace=False)] = 1
    return resultado


# ------------------------------------------------------------ drift de features
def fraccion_desde_evidently(resultado: dict) -> float:
    """Saca la fracción de variables con drift del resultado de Evidently (`snapshot.dict()`)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return float(metrica["value"]["share"])
    raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")


def hay_alerta(fraccion: float, umbral: float) -> bool:
    """Hay alerta de drift de features si la fracción de variables con drift SUPERA el umbral."""
    return fraccion > umbral


# ------------------------------------------------------------ drift de evento
def validar_lote(lote: pd.DataFrame, objetivo: str) -> None:
    """Rechaza el lote si no trae las etiquetas reales (la columna objetivo, sin vacíos)."""
    if objetivo not in lote.columns:
        raise ValueError(
            f"El lote no trae la columna '{objetivo}' (etiquetas reales): "
            "no se puede calcular el drift de evento. Espera a que lleguen las etiquetas."
        )
    if lote[objetivo].isna().any():
        raise ValueError(
            f"El lote tiene valores vacíos en '{objetivo}': no se puede calcular el drift de evento."
        )


def drift_de_target(referencia: pd.Series, lote: pd.Series, alfa: float) -> dict:
    """Compara la tasa de unos del objetivo entre entrenamiento y lote con una prueba chi-cuadrado."""
    ref = referencia.astype(int)
    act = lote.astype(int)
    tabla = [
        [int(ref.sum()), len(ref) - int(ref.sum())],
        [int(act.sum()), len(act) - int(act.sum())],
    ]
    _, p_valor, _, _ = chi2_contingency(tabla)
    return {
        "tasa_referencia": float(ref.mean()),
        "tasa_lote": float(act.mean()),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }
