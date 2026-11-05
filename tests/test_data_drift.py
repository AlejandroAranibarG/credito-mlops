"""Pruebas de data drift: deben estar VERDES con datos del mismo origen y ROJAS con datos derivados."""
import pytest

from datos import generar
from drift import estado_global, evaluar_data_drift


@pytest.fixture(scope="module")
def referencia():
    return generar(n=4000, seed=42)      # baseline de entrenamiento


def test_sin_deriva_estado_verde(referencia):
    lote = generar(n=1000, seed=1001)    # mismo origen que el entrenamiento
    informe = evaluar_data_drift(referencia, lote)
    print("\n", informe[["variable", "prueba", "estadistico", "p_valor", "estado"]])
    assert estado_global(informe) == "VERDE"


def test_tasa_de_falsas_alarmas_es_baja(referencia):
    """Con datos del mismo origen el umbral no debe disparar alarmas de forma habitual."""
    rojos = sum(estado_global(evaluar_data_drift(referencia, generar(n=1000, seed=s))) == "ROJO"
                for s in range(2000, 2100))
    print(f"\nfalsas alarmas: {rojos} de 100 lotes del mismo origen")
    assert rojos <= 10


def test_con_deriva_estado_rojo(referencia):
    lote = generar(n=1000, seed=7, data_drift=True)   # deriva inyectada deliberadamente
    informe = evaluar_data_drift(referencia, lote)
    print("\n", informe[["variable", "prueba", "estadistico", "p_valor", "estado"]])
    assert estado_global(informe) == "ROJO"


def test_la_deriva_se_detecta_en_las_variables_modificadas(referencia):
    lote = generar(n=1000, seed=7, data_drift=True)
    informe = evaluar_data_drift(referencia, lote).set_index("variable")
    for var in ["ingreso_mensual", "deuda_actual", "tipo_empleo"]:
        assert informe.loc[var, "estado"] == "ROJO", var
    # las variables que no se tocaron deben seguir verdes
    for var in ["edad", "plazo_meses", "monto_solicitado"]:
        assert informe.loc[var, "estado"] == "VERDE", var