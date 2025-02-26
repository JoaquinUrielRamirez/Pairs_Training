import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

class KalmanFilterReg():
    def __init__(self):
        self.x = np.array([1, 1])  # Initial Observation
        self.A = np.eye(2)  # Transition matrix
        self.Q = np.ones(2) * 10  # covariance matrix in estimations
        self.R = np.array([[1]]) * 100  # error in observations
        self.P = np.eye(2) * 10  # predicted error covariance matrix

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        C = np.array([[1, x]])  # Observation (1, 2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)  # Kalman Gain
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y - C @ self.x)


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


def prueba_cointegracion_johansen(data, activos, alpha=0.05):
    """
    Realiza la prueba de cointegración de Johansen para evaluar relaciones de cointegración.

    Parámetros:
    - data: DataFrame con precios de los activos.
    - activos: Lista de activos a analizar (mínimo 2).
    - alpha: Nivel de significancia (por defecto 0.05).

    Retorna:
    - Diccionario con los valores propios y vectores de cointegración.
    """
    resultados = {}

    # Extraer datos de los activos
    serie = data[activos].dropna()

    # Prueba de Johansen
    resultado_johansen = coint_johansen(serie, det_order=0, k_ar_diff=1)

    # Valores propios
    resultados['Eigenvalues'] = resultado_johansen.eig

    # Estadísticas de traza
    resultados['Trace Statistic'] = resultado_johansen.lr1
    resultados['Critical Values (90%, 95%, 99%)'] = resultado_johansen.cvt

    # Vectores de cointegración (beta)
    resultados['Cointegration Vectors (Beta)'] = resultado_johansen.evec

    # Determinar el número de relaciones de cointegración
    relaciones = sum(resultado_johansen.lr1 > resultado_johansen.cvt[:, 1])  # Comparar con el 95%
    resultados['Número de Cointegraciones'] = relaciones

    return resultados


def entrenar_vecm(data, activos, num_lags=1):
    """
    Entrena un modelo VECM para activos cointegrados.

    Parámetros:
    - data: DataFrame con precios de los activos.
    - activos: Lista de activos a analizar.
    - num_lags: Número de lags para el modelo.
    Retorna:
    - Modelo VECM entrenado y término de corrección de error.
    """
    # Extraer datos de los activos
    serie = data[activos].dropna()

    # Ajustar modelo VECM
    modelo_vecm = VECM(serie, k_ar_diff=num_lags, coint_rank=1)
    resultado = modelo_vecm.fit()

    # Obtener el término de corrección de error (mu)
    mu = resultado.alpha @ resultado.beta.T @ serie.T

    return resultado, mu.T


def aplicar_filtro_kalman_reg(series):
    """
    Aplica el Filtro de Kalman para estimar la relación dinámica entre dos activos.

    Parámetros:
    - series: DataFrame con precios de los activos (debe tener dos columnas).

    Retorna:
    - hedge_ratio: Serie con el hedge ratio dinámico sin valores NaN.
    """
    kf = KalmanFilterReg()
    hedge_ratios = []

    for index, row in series.iterrows():
        kf.predict()
        kf.update(row.iloc[1], row.iloc[0])  # y = activo 1, x = activo 2
        hedge_ratios.append(kf.x[1])

    hedge_ratio_series = pd.Series(hedge_ratios, index=series.index)

    # Eliminar valores NaN en lugar de rellenarlos
    hedge_ratio_series.dropna(inplace=True)

    return hedge_ratio_series


def backtest_estrategia_balanceada(data, señales, hedge_ratio, capital_inicial=1_000_000, comision=0.00125):
    """
    Realiza un backtest de la estrategia considerando margen, comisiones y capital inicial.

    Parámetros:
    - data: DataFrame con precios de los activos.
    - señales: DataFrame con señales de trading balanceadas.
    - hedge_ratio: Serie con el hedge ratio dinámico.
    - capital_inicial: Capital disponible para operar (por defecto $1,000,000 USD).
    - comision: Costo de transacción por operación (0.125%).

    Retorna:
    - DataFrame con la evolución del capital y el rendimiento acumulado.
    """
    capital = capital_inicial
    posicion = 0
    rendimiento = []
    capital_hist = []

    for i in range(len(señales)):
        if señales['Long'].iloc[i] == 1:
            size = (capital * 0.10) / data.iloc[i, 0]  # 10% del capital
            costo_transaccion = size * data.iloc[i, 0] * comision
            capital -= costo_transaccion
            posicion += size * hedge_ratio.iloc[i]

        elif señales['Short'].iloc[i] == 1:
            size = (capital * 0.10) / data.iloc[i, 0]  # 10% del capital
            costo_transaccion = size * data.iloc[i, 0] * comision
            capital -= costo_transaccion
            posicion -= size * hedge_ratio.iloc[i]

        # Calcular valor de la posición
        valor_posicion = posicion * data.iloc[i, 0]
        capital_actual = capital + valor_posicion
        rendimiento.append(capital_actual - capital_inicial)
        capital_hist.append(capital_actual)

    resultado = pd.DataFrame({'Capital': capital_hist, 'Rendimiento': rendimiento}, index=data.index)
    return resultado