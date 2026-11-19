"""
inferencia_flow.py
Pipeline de inferencia en Prefect, con drift emulado (actividad 3).

Pasos: cargar champion -> ingesta del lote (parte de prueba + drift emulado) -> predicción -> registro en MLflow

- Carga el modelo por su alias "champion" (models:/<nombre>@champion), nunca por número de versión.
- El lote sale de la parte de PRUEBA del dataset (misma división que el entrenamiento) y se le aplica
  el drift emulado con el nivel de configs/config.yaml (inferencia.nivel_drift). Ver src/monitoreo.py.
- Guarda una copia del lote puntuado en data/lotes/ (una por ejecución) y en data/lotes/lote_actual.csv,
  que es el que leen los flows de drift.
- Las predicciones se guardan en data/predicciones/prediccion_<fecha>.csv

Ejecución (PowerShell, desde la raíz del proyecto):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/inferencia_flow.py
"""

import os
import sys
from datetime import date, datetime
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402

from src.inferencia import armar_lote, nombre_archivo_lote, predecir_lote  # noqa: E402
from src.load_data import leer_config, load_data  # noqa: E402
from src.modelo import dividir  # noqa: E402
from src.monitoreo import columnas_afectadas  # noqa: E402


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
    """Toma un lote de la parte de prueba, le aplica el drift emulado y guarda una copia."""
    nivel = float(cfg["inferencia"]["nivel_drift"])
    objetivo = cfg["objetivo"]["nombre"]
    _, prueba = dividir(load_data(), cfg["objetivo"], cfg["division"])

    lote = armar_lote(
        prueba, cfg["inferencia"]["tamano_lote"], nivel, objetivo, cfg["division"]["random_state"]
    )
    print(f"Nivel de drift (config.yaml): {nivel}")
    print(f"Variables alteradas: {columnas_afectadas(nivel) or 'ninguna'}")
    print(f"Tasa de {objetivo} en el lote: {lote[objetivo].mean():.1%}")

    carpeta = RAIZ_PROYECTO / cfg["monitoreo"]["carpeta_lotes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    momento = datetime.now().strftime("%Y%m%d-%H%M%S")
    copia = carpeta / nombre_archivo_lote(nivel, momento)
    lote.to_csv(copia, index=False)
    lote.to_csv(RAIZ_PROYECTO / cfg["monitoreo"]["lote_actual"], index=False)
    print(f"Lote de {len(lote)} vacantes guardado en {copia} y en lote_actual.csv")
    return lote, str(copia)


@task(name="prediccion")
def prediccion(modelo, lote, cfg):
    """Calcula la probabilidad de salario alto y guarda el CSV con la fecha de hoy."""
    resultado = predecir_lote(modelo, lote, cfg["metricas"]["umbral"])

    carpeta = RAIZ_PROYECTO / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)

    print(f"Predicciones guardadas en {ruta}")
    print(f"Vacantes con salario alto predicho: {int(resultado['prediccion'].sum())} de {len(resultado)}")
    return str(ruta), len(resultado), float(resultado["probabilidad_salario_alto"].mean())


@task(name="registro-inferencia")
def registro_inferencia(version, run_id_entrenamiento, nivel, ruta, ruta_lote, filas, probabilidad_media):
    """Crea un run de inferencia en MLflow, conectado con la versión del modelo y el nivel de drift."""
    with mlflow.start_run(run_name=f"inferencia-drift-{nivel}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.set_tag("nivel_drift", nivel)
        mlflow.log_param("nivel_drift", nivel)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", probabilidad_media)
        mlflow.log_artifact(ruta)
        mlflow.log_artifact(ruta_lote)
    print("Run de inferencia registrado en MLflow")


@flow(name="inferencia-salario", log_prints=True)
def inferencia_flow():
    cfg = leer_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote, ruta_lote = ingesta_lote(cfg)
    ruta, filas, probabilidad_media = prediccion(modelo, lote, cfg)
    registro_inferencia(
        version, run_id_entrenamiento, cfg["inferencia"]["nivel_drift"],
        ruta, ruta_lote, filas, probabilidad_media,
    )


if __name__ == "__main__":
    inferencia_flow()
