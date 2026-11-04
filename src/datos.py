"""Generador de datos sinteticos de riesgo crediticio con control de drift."""
import numpy as np
import pandas as pd

NUMERICAS = ["ingreso_mensual", "deuda_actual", "antiguedad_laboral",
             "monto_solicitado", "plazo_meses", "edad", "num_creditos_previos"]
CATEGORICAS = ["tipo_empleo"]
FEATURES = NUMERICAS + CATEGORICAS
TARGET = "moroso"
TIPOS = ["dependiente", "independiente", "informal"]


def generar(n=4000, seed=42, data_drift=False, concept_drift=False):
    """Genera n solicitudes.

    data_drift=True    -> cambia la distribucion de las ENTRADAS (ingresos
                          menores, mas deuda, mas empleo informal).
    concept_drift=True -> cambia la REGLA que genera la etiqueta: las entradas
                          se ven igual, pero la relacion con 'moroso' cambia.
    """
    rng = np.random.default_rng(seed)

    if data_drift:
        ingreso = rng.gamma(4.0, 280.0, n) + 250
        deuda = rng.gamma(2.5, 1100.0, n)
        p_empleo = [0.30, 0.30, 0.40]
    else:
        ingreso = rng.gamma(4.0, 400.0, n) + 300
        deuda = rng.gamma(2.0, 900.0, n)
        p_empleo = [0.60, 0.25, 0.15]

    antig = rng.exponential(5.0, n).clip(0, 35)
    monto = rng.gamma(2.5, 2200.0, n) + 500
    plazo = rng.choice([6, 12, 18, 24, 36, 48, 60], n)
    edad = rng.integers(21, 66, n)
    previos = rng.poisson(1.5, n)
    empleo = rng.choice(TIPOS, n, p=p_empleo)

    cuota = monto / plazo
    carga = (deuda * 0.04 + cuota) / ingreso
    informal = (empleo == "informal").astype(float)

    if not concept_drift:
        score = (2.2 * carga - 0.06 * antig - 0.01 * (edad - 40) / 10
                 + 0.15 * informal - 0.03 * previos)
        umbral = 0.55
    else:
        # Nueva regla: la carga financiera pesa menos, la antiguedad y los
        # creditos previos cambian de signo, el empleo informal pesa mas.
        score = (0.9 * carga + 0.05 * antig + 0.12 * previos
                 + 0.45 * informal - 0.01 * (edad - 40) / 10)
        umbral = 1.0
    score = score + rng.normal(0, 0.35, n)

    df = pd.DataFrame({
        "ingreso_mensual": ingreso.round(2),
        "deuda_actual": deuda.round(2),
        "antiguedad_laboral": antig.round(1),
        "monto_solicitado": monto.round(2),
        "plazo_meses": plazo,
        "edad": edad,
        "num_creditos_previos": previos,
        "tipo_empleo": empleo,
    })
    df[TARGET] = (score > umbral).astype(int)
    return df


if __name__ == "__main__":
    for nombre, kw in [("base", {}), ("data_drift", {"data_drift": True}),
                       ("concept_drift", {"concept_drift": True})]:
        d = generar(**kw)
        print(f"{nombre:14s} filas={len(d)} morosos={d[TARGET].mean():.1%} "
              f"ingreso_medio={d.ingreso_mensual.mean():.0f}")