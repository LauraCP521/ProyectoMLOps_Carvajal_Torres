"""
drift_evento_flow.py
Pipeline de Prefect que detecta drift de EVENTO, es decir, de la variable objetivo (actividad 3).

Pasos: referencia (tasa de salario_alto en entrenamiento) -> evaluar el lote -> alerta

- Compara la tasa de salario_alto del último lote puntuado con la de entrenamiento (prueba chi-cuadrado).
- Rechaza con un mensaje claro los lotes que no traen las etiquetas reales (en producción pueden
  llegar con retraso).
- Emite una ALERTA si el p-valor es menor que monitoreo.alfa_evento.
- No necesita MLflow (sí Prefect, para ver los logs en la UI).

Ejecución (PowerShell, desde la raíz del proyecto, DESPUÉS de la inferencia):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/drift_evento_flow.py
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

from src.load_data import leer_config, load_data  # noqa: E402
from src.modelo import dividir  # noqa: E402
from src.monitoreo import drift_de_target, validar_lote  # noqa: E402


@task(name="referencia-objetivo")
def referencia(cfg):
    """Variable objetivo de los datos de entrenamiento."""
    entrenamiento, _ = dividir(load_data(), cfg["objetivo"], cfg["division"])
    return entrenamiento[cfg["objetivo"]["nombre"]]


@task(name="evaluar-evento")
def evaluar_evento(referencia_objetivo, cfg):
    """Compara la tasa del objetivo del lote con la de entrenamiento. Falla si el lote no trae etiquetas."""
    objetivo = cfg["objetivo"]["nombre"]
    ruta_lote = RAIZ_PROYECTO / cfg["monitoreo"]["lote_actual"]
    if not ruta_lote.exists():
        raise FileNotFoundError(f"No hay lote en {ruta_lote}. Ejecuta primero la inferencia.")
    lote = pd.read_csv(ruta_lote)
    validar_lote(lote, objetivo)

    resultado = drift_de_target(referencia_objetivo, lote[objetivo], cfg["monitoreo"]["alfa_evento"])
    print(
        f"Tasa de {objetivo}: entrenamiento {resultado['tasa_referencia']:.1%}, "
        f"lote {resultado['tasa_lote']:.1%} (p-valor {resultado['p_valor']:.4f})"
    )
    return resultado


@task(name="alerta-evento")
def alerta(resultado, alfa):
    """Emite una alerta (log de advertencia) si hay drift significativo del objetivo."""
    logger = get_run_logger()
    if resultado["drift"]:
        logger.warning(f"ALERTA: drift del objetivo (p-valor {resultado['p_valor']:.4f} < {alfa})")
        return True
    logger.info(f"Sin alerta: drift del objetivo (p-valor {resultado['p_valor']:.4f} >= {alfa})")
    return False


@flow(name="drift-evento", log_prints=True)
def drift_evento_flow():
    cfg = leer_config()
    resultado = evaluar_evento(referencia(cfg), cfg)
    return alerta(resultado, cfg["monitoreo"]["alfa_evento"])


if __name__ == "__main__":
    drift_evento_flow()
