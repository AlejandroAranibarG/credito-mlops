# Despliegue reproducible y observable de un modelo de riesgo crediticio

Proyecto final, módulo ML-Ops y Puesta en Producción. Autor: Willy Alejandro Aranibar Gonzales (trabajo individual).

Un modelo de clasificación (riesgo de mora) se entrena con **MLflow**, se registra con versionado y alias, se empaqueta en **Docker**, se despliega en **Kubernetes** con 3 réplicas y se monitorea con pruebas automatizadas de **data drift** y **concept drift**.

## Estructura

| Ruta | Contenido |
|---|---|
| `src/datos.py` | Generador de datos sintéticos con deriva controlable |
| `src/train.py` | Entrenamiento: 6 runs en MLflow, registro de versiones y alias `champion` |
| `src/api.py` | API FastAPI que carga `models:/credito-riesgo@champion` |
| `src/drift.py`, `src/monitor.py` | Detección de data drift (KS, PSI) y concept drift |
| `tests/` | Pruebas automatizadas de drift (pytest) y JSON de ejemplo para la API |
| `Dockerfile`, `requirements.txt` | Imagen de la API |
| `k8s/` | Deployment (3 réplicas) y Service |
| `docs/evidencias/` | Capturas de cada fase |

## Cómo reproducirlo

Requisitos: Python 3.12, Docker Desktop con Kubernetes activado.

```bash
conda create -n proyecto python=3.12 -y && conda activate proyecto
pip install mlflow==3.18.0 scikit-learn==1.9.1 pandas numpy scipy matplotlib pytest
```

1. **Servidor MLflow** (terminal aparte, se deja abierto):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination ./mlartifacts --serve-artifacts --host 0.0.0.0 --port 5000 --workers 1 \
  --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"
```

2. **Entrenar y registrar el modelo:** `python src/train.py`
3. **Construir la imagen:** `docker build -t credito-mlops:1.0 .`
4. **Desplegar en Kubernetes:** `kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml`
5. **Consumir el servicio:** `curl http://localhost:30090/modelo` y `POST http://localhost:30090/predecir` (ejemplos de cuerpo en `tests/bajo.json` y `tests/alto.json`)
6. **Pruebas de drift:** `python -m pytest -v -s` e informe con `python src/monitor.py`

El modelo no va dentro de la imagen: la API lo descarga del Model Registry de MLflow por alias al arrancar, así el servicio siempre sirve la versión marcada como `champion`.