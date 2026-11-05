"""Informe de monitoreo: data drift (verde/rojo) y concept drift (grafica temporal)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from datos import generar
from drift import (CAIDA_AUC, LOTES_SEGUIDOS, auc_referencia, cargar_modelo_champion,
                   estado_global, evaluar_data_drift, lote_de_alarma, simular_lotes)

pd.set_option("display.width", 160)
COLS = ["variable", "prueba", "estadistico", "p_valor", "umbral", "estado"]
SALIDA = Path("docs/evidencias")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    ref = generar(n=4000, seed=42)

    print("=" * 70, "\nDATA DRIFT 1: lote del mismo origen que el entrenamiento")
    inf = evaluar_data_drift(ref, generar(n=1000, seed=1001))
    print(inf[COLS].to_string(index=False), f"\n>>> ESTADO GLOBAL: {estado_global(inf)}")

    print("=" * 70, "\nDATA DRIFT 2: lote con deriva inyectada (ingresos menores, mas deuda, mas empleo informal)")
    inf = evaluar_data_drift(ref, generar(n=1000, seed=1001, data_drift=True))
    print(inf[COLS].to_string(index=False), f"\n>>> ESTADO GLOBAL: {estado_global(inf)}")

    print("=" * 70, "\nCONCEPT DRIFT: la regla que genera la etiqueta cambia en el lote 7")
    modelo, origen = cargar_modelo_champion()
    base = auc_referencia(modelo)
    lotes = simular_lotes(modelo)
    print(f"Modelo cargado desde: {origen} | AUC de referencia = {base:.4f} | "
          f"umbral de alarma = {base - CAIDA_AUC:.4f}")
    print(lotes.round(4).to_string(index=False))
    a0 = lote_de_alarma(lotes["auc"], base, retraso=0)
    a2 = lote_de_alarma(lotes["auc"], base, retraso=2)
    print(f"Alarma de reentrenamiento: lote {a0} (etiquetas inmediatas) | lote {a2} (etiquetas con 2 lotes de retraso)")

    print("Data drift sobre un lote con concept drift:",
          estado_global(evaluar_data_drift(ref, generar(n=1000, seed=1001, concept_drift=True))),
          "(las entradas no cambian, por eso no lo detecta)")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(lotes["lote"], lotes["auc"], marker="o", label="AUC por lote")
    ax.axhline(base, color="green", ls="--", label=f"AUC de referencia ({base:.3f})")
    ax.axhline(base - CAIDA_AUC, color="red", ls="--", label=f"Umbral de alarma ({base - CAIDA_AUC:.3f})")
    ax.axvline(6.5, color="gray", ls=":", label="Cambio de concepto (lote 7)")
    if a0:
        ax.axvline(a0, color="orange", ls="-.", label=f"Alarma de reentrenamiento (lote {a0})")
    ax.set_xlabel("Lote"); ax.set_ylabel("AUC"); ax.set_xticks(lotes["lote"])
    ax.set_title("Concept drift: degradacion del AUC en lotes sucesivos"); ax.legend(loc="lower left")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(SALIDA / "concept_drift_auc.png", dpi=130)
    print(f"Grafica guardada en {SALIDA / 'concept_drift_auc.png'}")


if __name__ == "__main__":
    main()