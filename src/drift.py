"""Deteccion de data drift y concept drift."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from datos import CATEGORICAS, FEATURES, NUMERICAS, TARGET, TIPOS, generar

# ---------------------------------------------------------------- DATA DRIFT
ALPHA = 0.05
ALPHA_KS = ALPHA / len(NUMERICAS)   # correccion de Bonferroni (7 pruebas KS)
UMBRAL_PSI = 0.20                   # convencion de la industria crediticia
EPS = 1e-4                          # evita log(0) en categorias vacias


def psi(ref: pd.Series, new: pd.Series, categorias=TIPOS) -> float:
    """Population Stability Index para una variable categorica."""
    p_ref = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).clip(lower=EPS)
    p_new = new.value_counts(normalize=True).reindex(categorias, fill_value=0).clip(lower=EPS)
    return float(((p_new - p_ref) * np.log(p_new / p_ref)).sum())


def evaluar_data_drift(ref: pd.DataFrame, lote: pd.DataFrame) -> pd.DataFrame:
    """Compara cada variable de entrada del lote contra el baseline de entrenamiento."""
    filas = []
    for col in NUMERICAS:
        ks = stats.ks_2samp(ref[col], lote[col])
        filas.append({"variable": col, "prueba": "KS",
                      "estadistico": round(float(ks.statistic), 4),
                      "p_valor": float(ks.pvalue),
                      "umbral": f"p < {ALPHA_KS:.4f}",
                      "alerta": bool(ks.pvalue < ALPHA_KS)})
    for col in CATEGORICAS:
        v = psi(ref[col], lote[col])
        filas.append({"variable": col, "prueba": "PSI",
                      "estadistico": round(v, 4), "p_valor": np.nan,
                      "umbral": f"PSI > {UMBRAL_PSI}", "alerta": bool(v > UMBRAL_PSI)})
    df = pd.DataFrame(filas)
    df["estado"] = np.where(df["alerta"], "ROJO", "VERDE")
    return df


def estado_global(informe: pd.DataFrame) -> str:
    return "ROJO" if informe["alerta"].any() else "VERDE"


# ------------------------------------------------------------- CONCEPT DRIFT
CAIDA_AUC = 0.05        # caida respecto al AUC de referencia que se considera relevante
LOTES_SEGUIDOS = 3      # lotes consecutivos por debajo del umbral para disparar la alarma


def cargar_modelo_champion(tracking_uri="http://localhost:5000",
                           nombre="credito-riesgo", alias="champion"):
    """Carga el modelo en produccion desde el Registry; si MLflow no esta disponible,
    reentrena localmente la misma configuracion (para poder correr las pruebas sin servidor)."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{tracking_uri}/health", timeout=3)   # falla rapido si no hay servidor
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(f"models:/{nombre}@{alias}"), "registry"
    except Exception:  # noqa: BLE001
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from train import construir
        df = generar(n=4000, seed=42)
        train, _ = train_test_split(df, test_size=0.3, random_state=42, stratify=df[TARGET])
        pipe = construir(GradientBoostingClassifier(random_state=42, n_estimators=150,
                                                    learning_rate=0.05, max_depth=3))
        pipe.fit(train[FEATURES], train[TARGET])
        return pipe, "local"


def auc_referencia(modelo) -> float:
    """AUC del modelo sobre 5000 solicitudes del mismo origen que el entrenamiento."""
    ref = generar(n=5000, seed=999)
    return float(roc_auc_score(ref[TARGET], modelo.predict_proba(ref[FEATURES])[:, 1]))


def simular_lotes(modelo, n_lotes=12, lote_deriva=7, tam=500, seed=100) -> pd.DataFrame:
    """Lotes sucesivos. Desde 'lote_deriva' cambia la regla que genera la etiqueta
    (las entradas siguen teniendo la misma distribucion)."""
    filas = []
    for i in range(1, n_lotes + 1):
        derivado = i >= lote_deriva
        df = generar(n=tam, seed=seed + i, concept_drift=derivado)
        proba = modelo.predict_proba(df[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        filas.append({"lote": i, "concept_drift": derivado,
                      "auc": float(roc_auc_score(df[TARGET], proba)),
                      "f1": float(f1_score(df[TARGET], pred)),
                      "tasa_pred_moroso": float(pred.mean()),
                      "tasa_real_moroso": float(df[TARGET].mean())})
    return pd.DataFrame(filas)


def lote_de_alarma(auc_serie, auc_base, caida=CAIDA_AUC, seguidos=LOTES_SEGUIDOS, retraso=0):
    """Primer lote (1-indexado) en el que se CONOCE la alarma de reentrenamiento.

    retraso = lotes que tardan en llegar las etiquetas reales: en el lote t solo se
    pueden evaluar los lotes <= t - retraso."""
    flags = [a < auc_base - caida for a in auc_serie]
    for t in range(1, len(flags) + 1):
        ultimo = t - retraso
        if ultimo >= seguidos and all(flags[ultimo - seguidos:ultimo]):
            return t
    return None