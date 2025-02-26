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



###
# ✅ 4. Aplicar VECM y generar señales de trading
c_vecm, mu = entrenar_vecm(data, tickers)
sena_tra = generar_senales_trading(mu)

# ✅ 5. Visualizar señales
visualizar_senales(mu, sena_tra)

# ✅ 6. Aplicar Filtro de Kalman (Corrigiendo el tamaño)
hedge_ratio_kalman = aplicar_filtro_kalman_reg(data)

# Asegurar que el hedge ratio tiene el mismo tamaño que data
hedge_ratio_kalman = hedge_ratio_kalman.reindex(data.index, method='ffill')

# ✅ 7. Ajustar estrategia balanceada con Hedge Ratio corregido
estr_bal = ajustar_estrategia_balanceada(sena_tra, hedge_ratio_kalman)
visualizar_trading_signals(data, estr_bal, tickers)
visualizar_senales(mu, estr_bal)

# ✅ 8. Verificación de tamaños antes del backtest
print(f"Tamaño de data: {len(data)}")
print(f"Tamaño de hedge_ratio: {len(hedge_ratio_kalman)}")
print(f"Tamaño de señales: {len(estr_bal)}")

# ✅ 9. Ejecutar Backtest (Activar cuando todo esté corregido)
resultado_backtest = backtest_estrategia_balanceada(data, estr_bal, hedge_ratio_kalman)
print(resultado_backtest)

# ✅ 10. Visualizar evolución del capital en el backtest
plt.figure(figsize=(12, 6))
plt.plot(resultado_backtest['Capital'], label='Capital Acumulado', color='blue')
plt.axhline(1_000_000, color='red', linestyle='dashed', label='Capital Inicial')
plt.xlabel('Fecha')
plt.ylabel('Capital ($)')
plt.title('Evolución del Capital en el Backtest')
plt.legend()
plt.grid()
plt.show()


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


def visualizar_trading_signals(data, señales, activos):
    """
    Genera un gráfico con los precios de ambos activos y las señales de compra/venta.

    Parámetros:
    - data: DataFrame con los precios de los activos.
    - señales: DataFrame con señales de trading (Long/Short).
    - activos: Lista con los nombres de los dos activos.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[activos[0]], label=activos[0], color='blue')
    plt.plot(data.index, data[activos[1]], label=activos[1], color='orange')

    # Marcar señales de compra y venta
    # Filtrar señales válidas (las que existen en data)
    long_signals_1 = señales[señales['Long_Activo1'] == 1].index
    short_signals_1 = señales[señales['Short_Activo1'] == 1].index
    long_signals_2 = señales[señales['Long_Activo2'] == 1].index
    short_signals_2 = señales[señales['Short_Activo2'] == 1].index

    plt.scatter(long_signals_1, data.loc[long_signals_1, activos[0]],
                color='green', marker='^', label=f'Compra {activos[0]}', alpha=1)
    plt.scatter(short_signals_1, data.loc[short_signals_1, activos[0]],
                color='red', marker='v', label=f'Venta {activos[0]}', alpha=1)
    plt.scatter(long_signals_2, data.loc[long_signals_2, activos[1]],
                color='blue', marker='^', label=f'Compra {activos[1]}', alpha=1)
    plt.scatter(short_signals_2, data.loc[short_signals_2, activos[1]],
                color='orange', marker='v', label=f'Venta {activos[1]}', alpha=1)

    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.title('Señales de Trading sobre los Precios de los Activos')
    plt.legend()
    plt.grid()
    plt.show()

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