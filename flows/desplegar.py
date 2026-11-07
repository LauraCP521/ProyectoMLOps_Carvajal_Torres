"""
desplegar.py
Registra en Prefect los dos despliegues:

- entrenamiento-salario/entrenamiento-manual: sin horario, se lanza a mano.
- inferencia-salario/inferencia-diaria: todos los días según despliegue.cron_inferencia
  y despliegue.zona_horaria de configs/config.yaml (por defecto 06:00 a. m., hora de Colombia).

Ejecución (PowerShell, desde la raíz del proyecto):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/desplegar.py

IMPORTANTE: deja esta terminal abierta. Mientras esté abierta, Prefect puede
lanzar las ejecuciones. Si la cierras, las ejecuciones programadas no se lanzan.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from prefect import serve  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.load_data import leer_config  # noqa: E402

if __name__ == "__main__":
    cfg = leer_config()

    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name="entrenamiento-manual",
        description="Entrena, evalúa y registra el modelo. Se lanza a mano.",
    )
    despliegue_inferencia = inferencia_flow.to_deployment(
        name="inferencia-diaria",
        schedules=[
            Cron(cfg["despliegue"]["cron_inferencia"], timezone=cfg["despliegue"]["zona_horaria"])
        ],
        description="Predice con el modelo champion una vez al día.",
    )

    print("Despliegues registrados. Deja esta terminal abierta (Ctrl + C para detener).")
    serve(despliegue_entrenamiento, despliegue_inferencia)
