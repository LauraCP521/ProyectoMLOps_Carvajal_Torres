"""
modelo.py
Lógica del modelo: variable objetivo, selección de variables, pipeline y métricas.

Este archivo NO usa Prefect ni MLflow: es la "receta" del modelo.
Así se puede probar sin tener los servicios encendidos (ver tests/test_modelo.py).
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_FEATURES = RAIZ_PROYECTO / "configs" / "features.yaml"

# Variables disponibles para el modelo.
# No se usan job_id (identificador) ni salary (de ella se calcula la respuesta).
NUMERICAS = [
    "years_experience",
    "skills_python",
    "skills_sql",
    "skills_ml",
    "skills_deep_learning",
    "skills_cloud",
    "job_posting_month",
    "job_posting_year",
    "job_openings",
]
CATEGORICAS = [
    "job_title",
    "company_size",
    "company_industry",
    "country",
    "remote_type",
    "experience_level",
    "education_level",
    "hiring_urgency",
]
VARIABLES = NUMERICAS + CATEGORICAS


def crear_objetivo(df: pd.DataFrame, cfg_objetivo: dict) -> pd.Series:
    """Devuelve 1 si la vacante tiene salario alto (salary >= umbral) y 0 si no."""
    salario = df[cfg_objetivo["columna_origen"]]
    return (salario >= cfg_objetivo["umbral_salario"]).astype(int)


def leer_features(ruta: Path = RUTA_FEATURES) -> dict:
    """Lee configs/features.yaml."""
    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def resolver_features(cfg_features: dict) -> tuple[str, list[str], list[str]]:
    """
    Convierte la selección de features.yaml en listas de columnas.
    Devuelve (alias, numericas, categoricas).
    """
    alias = str(cfg_features["seleccion"]).upper()

    if alias == "ALL":
        elegidas = VARIABLES
    elif alias == "NUMERIC":
        elegidas = NUMERICAS
    elif alias == "CUSTOM":
        elegidas = cfg_features.get("custom") or []
        desconocidas = [v for v in elegidas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(
                f"Estas variables de 'custom' no existen o no se pueden usar: {desconocidas}"
            )
    else:
        raise ValueError(f"seleccion debe ser ALL, NUMERIC o CUSTOM, no '{alias}'")

    if not elegidas:
        raise ValueError("La selección de features no tiene ninguna variable.")

    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return alias, numericas, categoricas


def construir_modelo(tipo: str, parametros: dict, numericas: list, categoricas: list) -> Pipeline:
    """Arma el pipeline: preprocesado (escalar números y codificar textos) + clasificador."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "random_forest":
        estimador = RandomForestClassifier(**parametros)
    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)
    else:
        raise ValueError(f"Modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))

    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def calcular_metricas(y_real, probabilidad, umbral: float) -> dict:
    """Métricas de clasificación a partir de la probabilidad de salario alto."""
    prediccion = (probabilidad >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_real, probabilidad)),
        "accuracy": float(accuracy_score(y_real, prediccion)),
        "precision": float(precision_score(y_real, prediccion, zero_division=0)),
        "recall": float(recall_score(y_real, prediccion, zero_division=0)),
        "f1": float(f1_score(y_real, prediccion, zero_division=0)),
    }
