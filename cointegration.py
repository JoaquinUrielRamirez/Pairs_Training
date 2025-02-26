import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

def analizar_cointegracion(data, activos, alpha=0.05):
    """
    Analiza si dos activos son estacionarios y si están cointegrados.

    Parámetros:
    - data: DataFrame con los precios de los activos.
    - activos: Lista con los nombres de los dos activos a analizar.
    - alpha: Nivel de significancia para las pruebas (por defecto 0.05).

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