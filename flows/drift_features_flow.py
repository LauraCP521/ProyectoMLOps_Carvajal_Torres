"""
drift_features_flow.py
Pipeline de Prefect que detecta drift de FEATURES con Evidently (actividad 3).

Pasos: referencia (datos de entrenamiento) -> calcular drift del lote -> alerta

- Compara cada variable de entrada del último lote puntuado (monitoreo.lote_actual) con los datos
  de entrenamiento, usando el preset DataDriftPreset de Evidently.
- Guarda un reporte HTML por lote en reports/ (el nombre incluye el nivel de drift y la fecha-hora).
- Emite una ALERTA si la fracción de variables con drift supera monitoreo.umbral_drift.
- No necesita MLflow (sí Prefect, para ver los logs en la UI).

Ejecución (PowerShell, desde la raíz del proyecto, DESPUÉS de la inferencia):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/drift_features_flow.py
"""

import sys
from datetime import datetime
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import pandas as pd  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

from src.load_data import leer_config, load_data  # noqa: E402
from src.modelo import VARIABLES, dividir  # noqa: E402
from src.monitoreo import fraccion_desde_evidently, hay_alerta  # noqa: E402


@task(name="referencia")
def referencia(cfg):
    """Variables de entrada de los datos de entrenamiento: la base contra la que se compara."""
    entrenamiento, _ = dividir(load_data(), cfg["objetivo"], cfg["division"])
    print(f"Referencia: {len(entrenamiento)} vacantes de entrenamiento, {len(VARIABLES)} variables")
    return entrenamiento[VARIABLES]


@task(name="calcular-drift")
def calcular_drift(datos_referencia, cfg):
    """Ejecuta Evidently, guarda el reporte HTML y devuelve la fracción de variables con drift."""
    ruta_lote = RAIZ_PROYECTO / cfg["monitoreo"]["lote_actual"]
    if not ruta_lote.exists():
        raise FileNotFoundError(f"No hay lote en {ruta_lote}. Ejecuta primero la inferencia.")
    lote = pd.read_csv(ruta_lote)
    nivel = float(lote["nivel_drift"].iloc[0]) if "nivel_drift" in lote.columns else -1
    print(f"Lote: {len(lote)} vacantes, nivel de drift {nivel}")

    resultado = Report([DataDriftPreset()]).run(
        reference_data=datos_referencia, current_data=lote[VARIABLES]
    )

    carpeta = RAIZ_PROYECTO / cfg["monitoreo"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    momento = datetime.now().strftime("%Y%m%d-%H%M%S")
    ruta_html = carpeta / f"drift_features_nivel_{nivel:.1f}_{momento}.html"
    resultado.save_html(str(ruta_html))
    print(f"Reporte de Evidently guardado en {ruta_html}")

    fraccion = fraccion_desde_evidently(resultado.dict())
    print(f"Variables con drift: {fraccion:.1%} de {len(VARIABLES)}")
    return fraccion


@task(name="alerta")
def alerta(fraccion, umbral):
    """Emite una alerta (log de advertencia) si la fracción de variables con drift supera el umbral."""
    logger = get_run_logger()
    if hay_alerta(fraccion, umbral):
        logger.warning(
            f"ALERTA: drift de features en {fraccion:.1%} de las variables (umbral {umbral:.0%})"
        )
        return True
    logger.info(f"Sin alerta: drift de features en {fraccion:.1%} de las variables (umbral {umbral:.0%})")
    return False


@flow(name="drift-features", log_prints=True)
def drift_features_flow():
    cfg = leer_config()
    fraccion = calcular_drift(referencia(cfg), cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"])


if __name__ == "__main__":
    drift_features_flow()
