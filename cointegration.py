import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

def analizar_cointegracion(data, activos, alpha=0.05):
    """
    Analiza si dos activos son estacionarios y si están cointegrados.
    Retorna:
    - Diccionario con los resultados de estacionariedad y cointegración.
    """
    resultados = {}
    for activo in activos:
        serie = data[activo].dropna()
        adf_test = adfuller(serie)
        p_value = adf_test[1]
        resultados[activo] = {
            "Estacionario": p_value < alpha,
            "ADF p-valor": p_value
        }
    # Si ambos activos son estacionarios, no tiene sentido hacer cointegración
    if resultados[activos[0]]["Estacionario"] and resultados[activos[1]]["Estacionario"]:
        resultados["Cointegración"] = "No aplica (ambos son estacionarios)"
        return resultados
    # Prueba de cointegración de Engle-Granger
    serie1, serie2 = data[activos[0]].dropna(), data[activos[1]].dropna()
    score, p_value, _ = coint(serie1, serie2)

    resultados["Cointegración"] = {
        "Cointegrado": p_value < alpha,
        "Coint p-valor": p_value
    }
    return resultados


def johansen_cointegration_test(df, det_order=0, k_ar_diff=1):
    """
    Realiza el test de cointegración de Johansen sobre el DataFrame df,
    que contiene las series (en columnas) a analizar.

    Parámetros:
      - df: DataFrame con las series en columnas.
      - det_order: Orden del determinista (0 para sin tendencia).
      - k_ar_diff: Número de diferencias a usar en el test.

    Imprime y retorna el objeto resultado del test.
    """
    result = coint_johansen(df, det_order, k_ar_diff)
    print("\n--- Test de Johansen ---")
    print("Eigenvalues:")
    print(result.eig)
    print("\nEigenvectors:")
    print(result.evec)
    print("\nEstadísticos de Rastreo (Trace):")
    print(result.lr1)
    print("\nValores críticos:")
    print(result.cvt)
    return result

