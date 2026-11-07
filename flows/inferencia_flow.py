"""
inferencia_flow.py
Pipeline de inferencia en Prefect (se programa para correr una vez al día en desplegar.py).

Pasos: cargar champion -> ingesta del lote -> predicción -> guardar CSV -> registrar en MLflow

- Carga el modelo por su alias "champion" (models:/<nombre>@champion), nunca por número de versión.
  Si el entrenamiento cambia el champion, la inferencia usa el nuevo sin tocar código.
- El lote simula vacantes nuevas: una muestra aleatoria del CSV.
- Las predicciones se guardan en data/predicciones/prediccion_<fecha>.csv

Ejecución (PowerShell, desde la raíz del proyecto):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/inferencia_flow.py
"""

import os
import sys
from datetime import date
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402

from src.load_data import leer_config, load_data  # noqa: E402


@task(name="cargar-champion")
def cargar_champion(cfg):
    """Busca en MLflow el modelo con alias champion y lo carga."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} @champion -> versión {version.version}")
    return modelo, version.version, version.run_id


@task(name="ingesta-lote")
def ingesta_lote(cfg):
    """Toma una muestra aleatoria del CSV que simula vacantes nuevas."""
    tamano = cfg["inferencia"]["tamano_lote"]
    df = load_data()
    lote = df.sample(n=tamano)
    print(f"Lote de {len(lote)} vacantes tomado al azar")
    return lote


@task(name="prediccion")
def prediccion(modelo, lote, cfg):
    """Calcula la probabilidad de salario alto y guarda el CSV con la fecha de hoy."""
    # El modelo usa las mismas columnas con las que se entrenó.
    columnas = list(modelo.feature_names_in_)
    probabilidad = modelo.predict_proba(lote[columnas])[:, 1]

    resultado = lote.copy()
    resultado["probabilidad_salario_alto"] = probabilidad.round(4)
    resultado["prediccion"] = (probabilidad >= cfg["metricas"]["umbral"]).astype(int)

    carpeta = RAIZ_PROYECTO / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)

    print(f"Predicciones guardadas en {ruta}")
    print(f"Vacantes con salario alto predicho: {int(resultado['prediccion'].sum())} de {len(resultado)}")
    return str(ruta), len(resultado), float(probabilidad.mean())


@task(name="registro-inferencia")
def registro_inferencia(version, run_id_entrenamiento, ruta, filas, probabilidad_media):
    """Crea un run de inferencia en MLflow, conectado con la versión del modelo usada."""
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", probabilidad_media)
        mlflow.log_artifact(ruta)
    print("Run de inferencia registrado en MLflow")


@flow(name="inferencia-salario", log_prints=True)
def inferencia_flow():
    cfg = leer_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta_lote(cfg)
    ruta, filas, probabilidad_media = prediccion(modelo, lote, cfg)
    registro_inferencia(version, run_id_entrenamiento, ruta, filas, probabilidad_media)


if __name__ == "__main__":
    inferencia_flow()
