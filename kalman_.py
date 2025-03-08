import numpy as np

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
###