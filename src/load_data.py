"""
load_data.py
Lee el dataset "AI & Data Science Job Market (2020-2026)" de Kaggle,
comprueba que tenga las columnas esperadas y lo devuelve listo para usar.

Fuente: https://www.kaggle.com/datasets/shree0910/ai-and-data-science-job-market-dataset-20202026

Uso desde la raíz del proyecto:
    poetry run python src/load_data.py
"""

from pathlib import Path

import pandas as pd
import yaml

# Carpeta raíz del proyecto (la carpeta que contiene a src/)
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ_PROYECTO / "configs" / "config.yaml"

# Columnas que DEBE tener el CSV. Si falta alguna, el script avisa.
COLUMNAS_ESPERADAS = [
    "job_id",
    "job_title",
    "company_size",
    "company_industry",
    "country",
    "remote_type",
    "experience_level",
    "years_experience",
    "education_level",
    "skills_python",
    "skills_sql",
    "skills_ml",
    "skills_deep_learning",
    "skills_cloud",
    "salary",
    "job_posting_month",
    "job_posting_year",
    "hiring_urgency",
    "job_openings",
]


def leer_config(ruta_config: Path = RUTA_CONFIG) -> dict:
    """Lee el archivo configs/config.yaml y lo devuelve como diccionario."""
    with open(ruta_config, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def load_data(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """
    Carga el CSV y valida sus columnas.

    Si no se indica ruta_csv, usa la ruta escrita en configs/config.yaml.
    """
    if ruta_csv is None:
        config = leer_config()
        ruta_csv = RAIZ_PROYECTO / config["data"]["path"]

    ruta_csv = Path(ruta_csv)

    # 1. ¿Existe el archivo?
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {ruta_csv}\n"
            "Descárgalo de Kaggle y guárdalo en la carpeta data/ "
            "(ver README)."
        )

    # 2. Leer el CSV
    df = pd.read_csv(ruta_csv)

    # 3. ¿Tiene todas las columnas esperadas?
    faltantes = [col for col in COLUMNAS_ESPERADAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Al dataset le faltan estas columnas: {faltantes}")

    # 4. ¿Tiene al menos una fila?
    if df.empty:
        raise ValueError("El dataset está vacío.")

    return df


if __name__ == "__main__":
    # Esto solo se ejecuta si corres el archivo directamente.
    datos = load_data()
    print("Dataset cargado correctamente.")
    print(f"Filas: {datos.shape[0]}  |  Columnas: {datos.shape[1]}")
    print(f"Valores vacíos en total: {int(datos.isna().sum().sum())}")
    print("\nPrimeras 5 filas:")
    print(datos.head())
