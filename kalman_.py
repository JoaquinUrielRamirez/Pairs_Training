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
def backtest_estrategia(mu, precios, hedge_ratios, capital_inicial=1_000_000, std_threshold=1.5, comision=0.00125):
    """
    Backtest de una estrategia de pairs trading con hedge ratio dinámico basado en Kalman Filter,
    asegurando que TODAS las posiciones abiertas se cierren cuando el spread regresa a su media.
    """
    señales = mu['signal']
    capital = capital_inicial
    posicion_por_trade = 0.1  # Cada trade usa 10% del capital disponible
    equity_curve = []  # Evolución del capital en cada iteración
    posiciones_abiertas = []
    trades_log = []

    mu_mean = mu['ECT'].mean()  # Asegurar que tomamos solo la columna ECT y convertimos a escalar

    for i in range(1, len(mu)):
        fecha = mu.index[i]
        signal = señales.iloc[i]
        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]
        hedge_ratio = hedge_ratios.iloc[i]  # Hedge Ratio dinámico de Kalman

        capital_trade = capital * posicion_por_trade

        if signal != 0:
            units_cvx = capital_trade / (1 + abs(hedge_ratio)) / price_cvx
            units_vlo = units_cvx * hedge_ratio

            nueva_posicion = {
                'Fecha': fecha,
                'Señal': 'Long CVX - Short VLO' if signal == 1 else 'Short CVX - Long VLO',
                'Unidades_CVX': units_cvx if signal == 1 else -units_cvx,
                'Unidades_VLO': -units_vlo if signal == 1 else units_vlo,
                'Precio_entrada_CVX': price_cvx,
                'Precio_entrada_VLO': price_vlo,
                'Hedge Ratio': hedge_ratio,
                'Capital_asignado': capital_trade if signal == 1 else 0,  # Capital solo se usa en LONG
                'Abierta': True
            }
            posiciones_abiertas.append(nueva_posicion)

        # Evaluar la evolución del capital con posiciones abiertas
        capital_temp = capital
        for posicion in posiciones_abiertas:
            if not posicion['Abierta']:
                continue

            pnl_cvx = (price_cvx - posicion['Precio_entrada_CVX']) * posicion['Unidades_CVX']
            pnl_vlo = (price_vlo - posicion['Precio_entrada_VLO']) * posicion['Unidades_VLO']

            capital_temp += pnl_cvx + pnl_vlo  # Capital solo cambia en LONG o cuando cierra un SHORT

        equity_curve.append(capital_temp)

        # Cierre de TODAS las posiciones si el spread regresa a la media
        if abs(mu['ECT'].iloc[i] - mu_mean) < 0.01 and posiciones_abiertas:
            for posicion in posiciones_abiertas:
                if not posicion['Abierta']:
                    continue

                hedge_ratio_original = posicion['Hedge Ratio']
                units_cvx_salida = posicion['Unidades_CVX']
                units_vlo_salida = posicion['Unidades_VLO']

                pnl_cvx = (price_cvx - posicion['Precio_entrada_CVX']) * units_cvx_salida
                pnl_vlo = (price_vlo - posicion['Precio_entrada_VLO']) * units_vlo_salida

                comision_cvx = abs(units_cvx_salida) * (posicion['Precio_entrada_CVX'] + price_cvx) * 0.5 * comision
                comision_vlo = abs(units_vlo_salida) * (posicion['Precio_entrada_VLO'] + price_vlo) * 0.5 * comision

                pnl_total = pnl_cvx + pnl_vlo - (comision_cvx + comision_vlo)

                if posicion['Señal'] == 'Short CVX - Long VLO':
                    capital += pnl_total  # El capital solo cambia al cerrar un SHORT

                posicion.update({
                    'Precio_salida_CVX': price_cvx,
                    'Precio_salida_VLO': price_vlo,
                    'PnL': pnl_total,
                    'Capital': capital,
                    'Fecha_cierre': fecha,
                    'Abierta': False
                })
                trades_log.append(posicion)

            # Una vez cerradas todas, vaciar la lista de posiciones abiertas
            posiciones_abiertas = []

    trades_log_df = pd.DataFrame(trades_log)
    backtest_result = pd.DataFrame(index=mu.index[1:], data={'Equity': equity_curve})

    return backtest_result, trades_log_df