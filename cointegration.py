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

def ols_regression_and_plot(series_dep, series_indep, dep_label="Y", indep_label="X"):
    """
    Realiza una regresión OLS de la serie dependiente (dep) sobre la serie independiente (indep)
    y grafica la dispersión junto con la línea de regresión.

    Parámetros:
      - series_dep: Serie de la variable dependiente.
      - series_indep: Serie de la variable independiente.
      - dep_label: Etiqueta para la variable dependiente (por defecto "Y").
      - indep_label: Etiqueta para la variable independiente (por defecto "X").
    """
    # Alinear las series por fechas (inner join)
    combined = pd.concat([series_dep, series_indep], axis=1, join='inner')
    combined.columns = [dep_label, indep_label]

    # Variables para la regresión
    X = combined[indep_label]
    y = combined[dep_label]
    X_const = sm.add_constant(X)  # Agrega una constante para el intercepto

    # Ajustar el modelo OLS
    model = sm.OLS(y, X_const).fit()
    print(model.summary())


class KalmanFilterReg:
    def __init__(self):
        self.x = np.array([1.0, 1.0])  # Estado inicial: [alpha, beta]
        self.A = np.eye(2)  # Matriz de transición (identidad)
        self.Q = np.ones(2) * 10  # Covarianza del proceso (ruido en el estado)
        self.R = np.array([[1]]) * 100  # Covarianza del error de observación
        self.P = np.eye(2) * 10  # Covarianza inicial del estado

    def predict(self):
        # Propaga la incertidumbre
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_val, y_val):
        """
        Observación: y_val = alpha + beta * x_val
        """
        C = np.array([[1, x_val]])  # Matriz de observación (1x2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)  # Ganancia de Kalman
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y_val - C @ self.x)


def run_kalman_filter_custom(log_x, log_y):
    """
    Aplica el filtro de Kalman en modo rolling para estimar alpha, beta y el spread dinámico,
    donde el modelo es: y_t = alpha_t + beta_t * x_t.
    Retorna un DataFrame con columnas ['alpha', 'beta', 'spread'] indexado por fecha.
    """
    # Alinear las series
    df = pd.concat([log_x, log_y], axis=1).dropna()
    df.columns = ['x', 'y']

    kf = KalmanFilterReg()
    alphas = []
    betas = []
    spread_dyn = []  # spread = y - (alpha + beta*x)

    for date, row in df.iterrows():
        x_val = row['x']
        y_val = row['y']
        kf.predict()
        kf.update(x_val, y_val)

        alpha, beta = kf.x
        alphas.append(alpha)
        betas.append(beta)
        spread_dyn.append(y_val - (alpha + beta * x_val))

    out = pd.DataFrame({
        'alpha': alphas,
        'beta': betas,
        'spread': spread_dyn
    }, index=df.index)

    return out


def generate_vecm_signals(log_df, det_order=0, k_ar_diff=1, threshold_sigma=1.5):
    """
    Ajusta un VECM a las series logarítmicas (DataFrame con dos columnas, ej. ['CVX_Log', 'VLO_Log']),
    calcula el Error Correction Term (ECT) y genera señales de trading basadas en umbrales de ±threshold_sigma desviaciones estándar.

    Retorna:
      - signals_df: DataFrame con columnas ['ECT', 'signal'] indexado por fecha.
      - vecm_res: el modelo VECM ajustado.
    """
    from statsmodels.tsa.vector_ar.vecm import VECM

    # Ajustar el VECM (determinista en la cointegración 'co')
    vecm_model = VECM(log_df, deterministic='co', k_ar_diff=k_ar_diff, coint_rank=1)
    vecm_res = vecm_model.fit()

    # Extraer el primer vector cointegrante
    beta = vecm_res.beta[:, 0]  # vector cointegrante
    # Si hay constante en la cointegración, beta puede tener un elemento extra
    if len(beta) > 2:
        const = beta[-1]
        beta_assets = beta[:-1]
    else:
        const = 0.0
        beta_assets = beta

    # Calcular ECT utilizando los datos en t-1 (shift)
    log_df_shift = log_df.shift(1).dropna()
    ect = log_df_shift.dot(beta_assets) + const

    # Calcular umbrales
    ect_mean = ect.mean()
    ect_std = ect.std()
    up_threshold = ect_mean + threshold_sigma * ect_std
    down_threshold = ect_mean - threshold_sigma * ect_std

    # Generar señales: -1 para venta (ECT alto), 1 para compra (ECT bajo), 0 para sin señal.
    signals = ect.apply(lambda x: -1 if x > up_threshold else (1 if x < down_threshold else 0))
    signals_df = pd.DataFrame({'ECT': ect, 'signal': signals})
    return signals_df, vecm_res
