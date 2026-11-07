"""
entrenamiento_flow.py
Pipeline de entrenamiento en Prefect.

Pasos: ingesta -> entrenamiento -> evaluación -> registro

- Cada ejecución crea un run en MLflow con el tag fase = entrenamiento.
- El modelo solo pasa a ser "champion" si:
    1) su AUC es mayor o igual que registro.umbral_auc, y
    2) supera el AUC del champion actual en al menos registro.mejora_minima.
  Si no, el run queda guardado en MLflow, pero el alias champion no cambia.

Ejecución (PowerShell, desde la raíz del proyecto):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    $env:PYTHONUTF8 = "1"
    poetry run python flows/entrenamiento_flow.py
"""

import os
import sys
from pathlib import Path

# Permite importar "src" aunque ejecutemos el archivo desde flows/
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import mlflow  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, task  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.load_data import leer_config, load_data  # noqa: E402
from src.modelo import (  # noqa: E402
    calcular_metricas,
    construir_modelo,
    crear_objetivo,
    leer_features,
    resolver_features,
)


# ---------------------------------------------------------------- 1. INGESTA
@task(name="ingesta")
def ingesta(cfg):
    """Carga el CSV, crea la variable objetivo y divide en entrenamiento y prueba."""
    df = load_data()
    print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")

    y = crear_objetivo(df, cfg["objetivo"])
    print(
        f"Objetivo '{cfg['objetivo']['nombre']}': salary >= {cfg['objetivo']['umbral_salario']} "
        f"-> {y.mean():.1%} de las vacantes tienen salario alto"
    )

    division = cfg["division"]
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=y,
    )
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------- 2. ENTRENAMIENTO
@task(name="entrenamiento")
def entrenamiento(X_train, y_train, cfg, cfg_features):
    """Elige las variables según features.yaml y entrena el modelo del config.yaml."""
    alias, numericas, categoricas = resolver_features(cfg_features)
    variables = numericas + categoricas
    print(f"Selección de variables: {alias} ({len(variables)} variables)")
    print(f"Variables: {variables}")

    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}
    print(f"Entrenando {tipo} con parámetros {parametros}...")

    modelo = construir_modelo(tipo, parametros, numericas, categoricas)
    modelo.fit(X_train[variables], y_train)
    print("Modelo entrenado")
    return modelo, alias, variables


# ------------------------------------------------------------- 3. EVALUACIÓN
@task(name="evaluacion")
def evaluacion(modelo, X_test, y_test, variables, cfg):
    """Calcula las métricas con los datos de prueba (datos que el modelo no vio)."""
    probabilidad = modelo.predict_proba(X_test[variables])[:, 1]
    metricas = calcular_metricas(y_test, probabilidad, cfg["metricas"]["umbral"])
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")
    return metricas


# --------------------------------------------------------------- 4. REGISTRO
@task(name="registro")
def registro(modelo, X_test, variables, alias, metricas, cfg):
    """Guarda el run en MLflow y decide si el modelo pasa a ser champion."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    client = MlflowClient()
    nombre_modelo = cfg["registro"]["nombre_modelo"]
    auc = metricas["auc"]

    # 4.1 Guardar el run (siempre, aunque el modelo no llegue a champion)
    with mlflow.start_run(run_name=cfg["mlflow"]["nombre_run"]) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params(
            {
                "modelo": cfg["modelo"]["tipo"],
                **(cfg["modelo"].get("parametros") or {}),
                "features": alias,
                "n_features": len(variables),
                "semilla": cfg["division"]["random_state"],
                "umbral_salario": cfg["objetivo"]["umbral_salario"],
            }
        )
        mlflow.log_metrics(metricas)
        ejemplo = X_test[variables].head(3)
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict_proba(ejemplo)[:, 1]),
            input_example=ejemplo,
            # Los modelos de árboles los marca skops como "no confiables" por defecto.
            # Son modelos entrenados por nosotras, así que confiamos en ese tipo.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
        print(f"Run guardado en MLflow: {run_id}")

        # 4.2 ¿Supera el AUC mínimo?
        minimo = cfg["registro"]["umbral_auc"]
        if auc < minimo:
            print(f"AUC {auc:.4f} por debajo del mínimo {minimo}: el champion no cambia")
            mlflow.set_tag("resultado_registro", "bajo_minimo")
            return None

        # 4.3 ¿Supera al champion actual?
        try:
            champion = client.get_model_version_by_alias(nombre_modelo, "champion")
        except MlflowException:
            champion = None  # todavía no hay champion

        if champion is not None:
            auc_champion = client.get_run(champion.run_id).data.metrics["auc"]
            mejora = cfg["registro"]["mejora_minima"]
            print(f"Champion actual: versión {champion.version} con AUC {auc_champion:.4f}")
            if auc < auc_champion + mejora:
                print(
                    f"AUC {auc:.4f} no supera al champion en al menos {mejora}: "
                    "el champion no cambia"
                )
                mlflow.set_tag("resultado_registro", "no_supera_champion")
                return None
        else:
            print("Todavía no hay champion: este modelo será el primero")

        # 4.4 Registrar nueva versión y moverle el alias champion
        version = mlflow.register_model(info.model_uri, nombre_modelo)
        client.set_registered_model_alias(nombre_modelo, "champion", version.version)
        client.set_model_version_tag(nombre_modelo, version.version, "estado", "aprobado")
        mlflow.set_tag("resultado_registro", "nuevo_champion")
        print(f"Versión {version.version} registrada como champion (AUC {auc:.4f})")
        return version.version


# -------------------------------------------------------------------- FLOW
@flow(name="entrenamiento-salario", log_prints=True)
def entrenamiento_flow():
    cfg = leer_config()
    cfg_features = leer_features()

    X_train, X_test, y_train, y_test = ingesta(cfg)
    modelo, alias, variables = entrenamiento(X_train, y_train, cfg, cfg_features)
    metricas = evaluacion(modelo, X_test, y_test, variables, cfg)
    version = registro(modelo, X_test, variables, alias, metricas, cfg)

    if version is None:
        print("Resultado: run guardado, el champion no cambió")
    else:
        print(f"Resultado: nuevo champion = versión {version}")
    return version


if __name__ == "__main__":
    entrenamiento_flow()
