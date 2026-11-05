"""Pruebas de concept drift: la relacion entradas -> etiqueta cambia y la metrica cae."""
import pytest

from datos import generar
from drift import (CAIDA_AUC, auc_referencia, cargar_modelo_champion,
                   estado_global, evaluar_data_drift, lote_de_alarma, simular_lotes)


@pytest.fixture(scope="module")
def modelo():
    m, origen = cargar_modelo_champion()
    print(f"\n[modelo cargado desde: {origen}]")
    return m


@pytest.fixture(scope="module")
def lotes(modelo):
    return simular_lotes(modelo)


def test_auc_estable_antes_de_la_deriva(modelo, lotes):
    base = auc_referencia(modelo)
    previos = lotes[~lotes["concept_drift"]]
    assert (previos["auc"] > base - CAIDA_AUC).all()


def test_auc_cae_despues_de_la_deriva(modelo, lotes):
    base = auc_referencia(modelo)
    derivados = lotes[lotes["concept_drift"]]
    assert (derivados["auc"] < base - CAIDA_AUC).all()


def test_alarma_de_reentrenamiento_se_dispara(modelo, lotes):
    base = auc_referencia(modelo)
    sin_retraso = lote_de_alarma(lotes["auc"], base, retraso=0)
    con_retraso = lote_de_alarma(lotes["auc"], base, retraso=2)
    assert sin_retraso == 9                  # deriva desde el lote 7 + 3 lotes seguidos
    assert con_retraso == sin_retraso + 2    # el retraso de etiquetas retrasa la deteccion


def test_data_drift_no_ve_el_concept_drift(modelo):
    """Las entradas son identicas: las pruebas de data drift siguen VERDES aunque el modelo falle."""
    ref = generar(n=4000, seed=42)
    lote = generar(n=1000, seed=1001, concept_drift=True)
    assert estado_global(evaluar_data_drift(ref, lote)) == "VERDE"