"""
Tests de DATOS: esquema, nulos, rangos y valores permitidos del dataset.

Ejecución (desde la raíz del proyecto):
    poetry run pytest tests/test_datos.py
"""

import pandas as pd
import pytest

from src.load_data import COLUMNAS_ESPERADAS, load_data

BINARIAS = ["skills_python", "skills_sql", "skills_ml", "skills_deep_learning", "skills_cloud"]

VALORES_PERMITIDOS = {
    "job_title": {"AI Engineer", "Business Analyst", "Data Analyst", "Data Engineer",
                  "Data Scientist", "Machine Learning Engineer"},
    "company_size": {"Enterprise", "MNC", "Medium", "Startup"},
    "company_industry": {"E-commerce", "Education", "Finance", "Healthcare", "Retail", "Technology"},
    "country": {"Australia", "Canada", "Germany", "India", "Singapore", "UK", "USA"},
    "remote_type": {"Hybrid", "Onsite", "Remote"},
    "experience_level": {"Entry", "Mid", "Senior"},
    "education_level": {"Bachelor", "Master", "PhD"},
    "hiring_urgency": {"High", "Low", "Medium"},
}


@pytest.fixture(scope="module")
def df():
    return load_data()


# ----------------------------------------------------------------- esquema
def test_tiene_exactamente_las_columnas_esperadas(df):
    assert list(df.columns) == COLUMNAS_ESPERADAS


def test_columnas_numericas_son_enteras(df):
    numericas = ["years_experience", "salary", "job_posting_month", "job_posting_year", "job_openings"]
    for columna in numericas + BINARIAS:
        assert pd.api.types.is_integer_dtype(df[columna]), columna


def test_job_id_es_unico(df):
    assert df["job_id"].is_unique


def test_load_data_rechaza_csv_sin_columnas(tmp_path):
    ruta = tmp_path / "incompleto.csv"
    pd.DataFrame({"job_id": [1], "salary": [100000]}).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="faltan"):
        load_data(ruta)


# ------------------------------------------------------------------- nulos
def test_no_hay_nulos(df):
    assert int(df.isna().sum().sum()) == 0


# ------------------------------------------------------------------ rangos
def test_binarias_solo_tienen_0_y_1(df):
    for columna in BINARIAS:
        assert set(df[columna].unique()) <= {0, 1}, columna


def test_rangos_razonables(df):
    assert df["salary"].between(10_000, 1_000_000).all()
    assert df["years_experience"].between(0, 50).all()
    assert df["job_posting_month"].between(1, 12).all()
    assert df["job_posting_year"].between(2020, 2026).all()
    assert (df["job_openings"] >= 1).all()


def test_categoricas_solo_tienen_valores_conocidos(df):
    for columna, permitidos in VALORES_PERMITIDOS.items():
        assert set(df[columna].unique()) <= permitidos, columna
