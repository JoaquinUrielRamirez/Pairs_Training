import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from se_trading import generate_signals_with_kalman, graficar_estrategia_y_ect, backtest_estrategia, calcular_metrica_backtest, graficar_equity_curve, graficar_activos_vs_estrategia, graficar_spread_trading
from cointegration import graficar_hedge_ratios, analizar_cointegracion, johansen_cointegration_test, generate_vecm_signals, run_kalman_filter_custom, ols_regression_and_plot

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
graficar_spread_trading(vecm_signals['ECT'])
print("\nSeñales (primeras 10 filas):")
print(vecm_signals.head(10))
print(len(vecm_signals))

#Kalman y VECM
graficar_hedge_ratios(kalman_results['beta'], vecm_res.beta[0, 0])

# Visualización de la estrategia de trading
graficar_estrategia_y_ect(norm, vecm_signals)

# Backtest de la estrategia con cierre de posiciones
capital_inicial = 1_000_000  # USD
comision = 0.00125

backtest_result, trades = backtest_estrategia(vecm_signals['ECT'], data, capital_inicial, comision=comision)

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

#Analisis de Trades
trades.to_excel('trades.xlsx', index=False)

if __name__ == '__main__':
    print('Ejecución completa')

