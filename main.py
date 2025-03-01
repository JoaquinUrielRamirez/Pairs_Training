import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from se_trading import generar_senales_trading, backtest_estrategia, calcular_metrica_backtest, graficar_equity_curve, graficar_activos_vs_estrategia, graficar_spread_trading
from cointegration import analizar_cointegracion, johansen_cointegration_test, generate_vecm_signals, run_kalman_filter_custom, ols_regression_and_plot

# Configuración de tickers y fechas
tickers = ['CVX', 'VLO']
start_date = "2015-08-22"

data = yf.download(tickers, start=start_date)["Open"]
data = pd.DataFrame(data).dropna()
norm = (data - data.mean()) / data.std()

# Verificación de cointegración y correlación
resultados = analizar_cointegracion(data, tickers)
print(resultados)

cor = data.corr()
print(f"Correlación entre {tickers[0]} y {tickers[1]}:", cor[tickers[0]][tickers[1]])
c_johansen = johansen_cointegration_test(data)
print(c_johansen)

# Realizar regresión OLS
print("\nResultados de la regresión OLS entre CVX y VLO:")
ols_regression_and_plot(data['VLO'], data['CVX'])

# Aplicar el filtro de Kalman para estimar el Hedge Ratio
log_data = np.log(data)
kalman_results = run_kalman_filter_custom(log_data['CVX'], log_data['VLO'])
print("\nHedge Ratio estimado con Kalman (últimos valores):")
print(kalman_results.tail())

# Agregar el Hedge Ratio al DataFrame
data['hedge_ratio'] = kalman_results['beta']

# Generar señales con VECM
vecm_signals, vecm_res = generate_vecm_signals(log_data, det_order=0, k_ar_diff=1, threshold_sigma=1.5)
print("\nSeñales (primeras 10 filas):")
print(vecm_signals.head(10))
print(len(vecm_signals))

# Backtest de la estrategia con cierre de posiciones
capital_inicial = 1_000_000  # USD
comision = 0.00125

backtest_result, trades = backtest_estrategia(vecm_signals['ECT'], data, capital_inicial, comision=comision)

# Visualización de la estrategia de trading
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,8))
ax1.plot(norm.index, norm['CVX'], label='CVX', color='blue')
ax1.plot(norm.index, norm['VLO'], label='VLO', color='black')

# Marcar señales sobre los precios normalizados
short_signals = vecm_signals[vecm_signals['signal'] == -1]
long_signals = vecm_signals[vecm_signals['signal'] == 1]

ax1.scatter(short_signals.index, norm['CVX'].reindex(short_signals.index), marker='v', color='red', s=100, label='Short CVX')
ax1.scatter(short_signals.index, norm['VLO'].reindex(short_signals.index), marker='^', color='green', s=100, label='Long VLO')
ax1.scatter(long_signals.index, norm['VLO'].reindex(long_signals.index), marker='v', color='red', s=100, label='Short VLO')
ax1.scatter(long_signals.index, norm['CVX'].reindex(long_signals.index), marker='^', color='green', s=100, label='Long CVX')

ax1.set_title("Comparación CVX vs VLO (10 años) + Señales Pairs Trading")
ax1.set_ylabel("Precio Normalizado")
ax1.grid(True)
ax1.legend()

# Graficar el ECT y las señales
ax2.plot(vecm_signals.index, vecm_signals['ECT'], label='ECT', color='purple')
ax2.axhline(vecm_signals['ECT'].mean() + 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='+1.5 Sigma')
ax2.axhline(vecm_signals['ECT'].mean() - 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='-1.5 Sigma')
ax2.axhline(vecm_signals['ECT'].mean(), color='red', linestyle='--', label='ECT Mean')
ax2.scatter(short_signals.index, short_signals['ECT'], marker='v', color='red', s=100, label='Señal Venta')
ax2.scatter(long_signals.index, long_signals['ECT'], marker='^', color='green', s=100, label='Señal Compra')
ax2.set_title("VECM: ECT y Señales de Trading (±1.5σ)")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("ECT")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Visualizar resultado del backtest
trades = trades.dropna()
metricas_df = calcular_metrica_backtest(trades, backtest_result)
# Mostrar métricas
print("Métricas del Backtest:")
print(metricas_df.to_string(index=False))

# Graficar curva de capital
graficar_equity_curve(backtest_result, trades)

# Grafico de Activos Originales vs Estrategia de Pares
graficar_activos_vs_estrategia(data, backtest_result, trades, vecm_signals['ECT'])
graficar_spread_trading(vecm_signals['ECT'])
#Analisis de Trades
trades.to_excel('trades.xlsx', index=False)

if __name__ == '__main__':
    print('Ejecución completa')

