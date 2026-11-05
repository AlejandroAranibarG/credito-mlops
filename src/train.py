"""Entrenamiento con MLflow: 6 runs comparables + registro de versiones + alias."""
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from datos import CATEGORICAS, FEATURES, TARGET, generar

TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "riesgo-crediticio"
MODEL_NAME = "credito-riesgo"
ALIAS = "champion"
SEED = 42
TEST_SIZE = 0.3

# (nombre del run, clase del modelo, hiperparametros)
CONFIGS = [
    ("rf_baseline_d3", RandomForestClassifier, dict(n_estimators=50, max_depth=3)),
    ("rf_d5", RandomForestClassifier, dict(n_estimators=100, max_depth=5)),
    ("rf_d8", RandomForestClassifier, dict(n_estimators=150, max_depth=8, min_samples_leaf=5)),
    ("rf_d12", RandomForestClassifier, dict(n_estimators=200, max_depth=12, min_samples_leaf=3)),
    ("gb_lr005", GradientBoostingClassifier, dict(n_estimators=150, learning_rate=0.05, max_depth=3)),
    ("gb_lr01", GradientBoostingClassifier, dict(n_estimators=200, learning_rate=0.1, max_depth=3)),
]


def construir(clf):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = generar(n=4000, seed=SEED)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET])
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # Baseline de referencia para las pruebas de drift
    Path("data").mkdir(exist_ok=True)
    train.to_csv("data/referencia.csv", index=False)

    resultados = []
    for nombre, clase, params in CONFIGS:
        with mlflow.start_run(run_name=nombre) as run:
            pipe = construir(clase(random_state=SEED, **params))
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = pipe.predict(X_test)
            metricas = {
                "auc": roc_auc_score(y_test, proba),
                "f1": f1_score(y_test, pred),
                "accuracy": accuracy_score(y_test, pred),
            }
            mlflow.log_params({"modelo": clase.__name__, "seed": SEED,
                               "test_size": TEST_SIZE, "n_train": len(train), **params})
            mlflow.log_metrics(metricas)
            info = mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_train, pipe.predict(X_train)),
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )
            resultados.append((nombre, run.info.run_id, info.model_uri, metricas))
            print(f"{nombre:16s} auc={metricas['auc']:.4f} f1={metricas['f1']:.4f}")

    # Registro: baseline como v1 y el mejor (por AUC) como v2
    mejor = max(resultados, key=lambda r: r[3]["auc"])
    base = resultados[0]
    client = MlflowClient()
    if base[1] != mejor[1]:
        v_base = mlflow.register_model(base[2], MODEL_NAME)
        print(f"Registrada version {v_base.version}: {base[0]} (baseline)")
    v_mejor = mlflow.register_model(mejor[2], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, v_mejor.version)
    print(f"Registrada version {v_mejor.version}: {mejor[0]} (run {mejor[1]})")
    print(f"Alias '{ALIAS}' -> version {v_mejor.version}")


if __name__ == "__main__":
    main()