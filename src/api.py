"""API de inferencia: carga el modelo desde el Model Registry de MLflow por alias."""
import os
import socket
import time
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credito-riesgo")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
POD_NAME = os.getenv("POD_NAME", socket.gethostname())

FEATURES = ["ingreso_mensual", "deuda_actual", "antiguedad_laboral",
            "monto_solicitado", "plazo_meses", "edad",
            "num_creditos_previos", "tipo_empleo"]


def cargar_modelo(intentos=20, espera=3):
    """Carga models:/<nombre>@<alias>. Reintenta por si MLflow aun no responde."""
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    for i in range(1, intentos + 1):
        try:
            modelo = mlflow.sklearn.load_model(uri)
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            print(f"[api] modelo cargado: {uri} -> version {mv.version} (run {mv.run_id})")
            return modelo, mv
        except Exception as e:  # noqa: BLE001
            print(f"[api] intento {i}/{intentos} fallo: {str(e)[:150]}")
            time.sleep(espera)
    raise RuntimeError(f"No se pudo cargar {uri} desde {TRACKING_URI}")


modelo, version = cargar_modelo()

app = FastAPI(title="API de Riesgo Crediticio", version="2.0")


class Solicitante(BaseModel):
    """Contrato de entrada con validaciones de negocio."""
    ingreso_mensual: float = Field(..., gt=0, description="Ingreso mensual (USD)")
    deuda_actual: float = Field(..., ge=0, description="Deuda actual (USD)")
    antiguedad_laboral: float = Field(..., ge=0, le=60, description="Anos en el empleo")
    monto_solicitado: float = Field(..., gt=0, description="Monto solicitado (USD)")
    plazo_meses: int = Field(..., gt=0, le=120, description="Plazo en meses")
    edad: int = Field(..., ge=18, le=100, description="Edad en anos")
    num_creditos_previos: int = Field(..., ge=0, description="Creditos previos")
    tipo_empleo: Literal["dependiente", "independiente", "informal"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/modelo")
def info_modelo():
    return {"nombre": MODEL_NAME, "alias": MODEL_ALIAS,
            "version": version.version, "run_id": version.run_id, "pod": POD_NAME}


@app.post("/predecir")
def predecir(s: Solicitante):
    X = pd.DataFrame([s.model_dump()])[FEATURES]
    prob = float(modelo.predict_proba(X)[0][1])
    moroso = int(prob >= 0.5)
    return {"moroso": moroso,
            "probabilidad_mora": round(prob, 4),
            "decision": "RECHAZAR" if moroso else "APROBAR",
            "modelo_version": version.version,
            "pod": POD_NAME}