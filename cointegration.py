import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from sklearn.linear_model import LinearRegression


def analizar_cointegracion(data, activos, alpha=0.05):
    """
    Analiza si dos activos son estacionarios y si están cointegrados.
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

    serie1, serie2 = data[activos[0]].dropna(), data[activos[1]].dropna()
    score, p_value, _ = coint(serie1, serie2)

    resultados["Cointegración"] = {
        "Cointegrado": p_value < alpha,
        "Coint p-valor": p_value
    }
    return resultados


def johansen_cointegration_test(df, det_order=0, k_ar_diff=1):
    """
    Realiza el test de cointegración de Johansen sobre el DataFrame df.
    """
    result = coint_johansen(df, det_order, k_ar_diff)
    return result


def generate_vecm_signals(data, threshold_sigma=1.5):
    """
    Genera señales de trading basadas en la cointegración usando el primer eigenvector de Johansen.
    """
    johansen_result = johansen_cointegration_test(data)
    beta = johansen_result.evec[:, 0]  # Primer eigenvector

    ect = data.dot(beta)
    ect_mean = ect.mean()
    ect_std = ect.std()

    up_threshold = ect_mean + threshold_sigma * ect_std
    down_threshold = ect_mean - threshold_sigma * ect_std

    signals = ect.apply(lambda x: -1 if x > up_threshold else (1 if x < down_threshold else 0))

    signals_df = pd.DataFrame({'ECT': ect, 'signal': signals})
    return signals_df


def ols_regression_and_plot(series_dep, series_indep, dep_label="VLO", indep_label="CVX"):

    combined = pd.concat([series_dep, series_indep], axis=1).dropna()
    combined.columns = [dep_label, indep_label]

    X = combined[[indep_label]].values.reshape(-1, 1)  # Variable independiente
    y = combined[dep_label].values  # Variable dependiente

    # Ajustar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Obtener coeficientes
    intercept = model.intercept_
    slope = model.coef_[0]
    r2 = model.score(X, y)

    print(f"Intercepto (α): {intercept:.4f}")
    print(f"Coeficiente (β - Hedge Ratio): {slope:.4f}")
    print(f"R²: {r2:.4f}")

    # Graficar los datos y la línea de regresión
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, label='Datos')
    plt.plot(X, model.predict(X), color='red', label='Regresión OLS')
    plt.xlabel(indep_label)
    plt.ylabel(dep_label)
    plt.title(f'Regresión OLS: {dep_label} vs {indep_label}')
    plt.legend()
    plt.grid(True)
    plt.show()


class KalmanFilterReg:
    """
    Implementación de un Filtro de Kalman desde cero para estimar el Hedge Ratio.
    """

    def __init__(self):
        self.x = np.array([1.0, 1.0])  # Estado inicial: [alpha, beta]
        self.A = np.eye(2)  # Matriz de transición (identidad)
        self.Q = np.ones(2) * 10  # Covarianza del proceso (ruido en el estado)
        self.R = np.array([[1]]) * 100  # Covarianza del error de observación
        self.P = np.eye(2) * 10  # Covarianza inicial del estado

    def predict(self):
        """ Propaga la incertidumbre del modelo de Kalman. """
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_val, y_val):
        """
        Actualiza el estado basado en la observación: y_val = alpha + beta * x_val
        """
        C = np.array([[1, x_val]])  # Matriz de observación (1x2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)  # Ganancia de Kalman
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y_val - C @ self.x)


def run_kalman_filter_custom(x, y):
    """
    Aplica el filtro de Kalman en modo rolling para estimar alpha, beta y el spread dinámico.
    """
    df = pd.concat([x, y], axis=1).dropna()
    df.columns = ['x', 'y']

    kf = KalmanFilterReg()
    alphas = []
    betas = []
    spread_dyn = []

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
