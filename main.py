import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from se_trading import backtest_estrategia, graficar_equity_curve, graficar_activos_vs_estrategia, graficar_spread_trading, backtest_estrategia_control_riesgo
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
print(f"Correlation beetwen {tickers[0]} and {tickers[1]}:", cor[tickers[0]][tickers[1]])
c_johansen = johansen_cointegration_test(data)
print(c_johansen)

# Realizar regresión OLS
print("\nResultados de la regresión OLS entre CVX y VLO:")
ols_regression_and_plot(data['VLO'], data['CVX'])

# Aplicar el filtro de Kalman para estimar el Hedge Ratio
kalman_results = run_kalman_filter_custom(data['CVX'], data['VLO'])
print(kalman_results['beta'].plot(label='Dinamic Hedge Ratio '))

# Generar señales con VECM usando Johansen y spread normalizado
vecm_signals = generate_vecm_signals(data, threshold_sigma=1.5)
graficar_spread_trading(vecm_signals['ECT'])
print("\nSeñales (primeras 10 filas):")
print(vecm_signals.head(10))
print(len(vecm_signals))

# Backtest de la estrategia con Hedge Ratio dinámico
capital_inicial = 1_000_000  # USD
comision = 0.00125

equity_df, trades = backtest_estrategia(data, vecm_signals, kalman_results['beta'], capital_inicial, comision)

graficar_equity_curve(equity_df, trades)
graficar_activos_vs_estrategia(data, equity_df, trades, vecm_signals)

trades = trades.dropna()

# Guardar análisis de trades
trades.to_excel('trades.xlsx', index=False)
equity_df.to_excel('portafolio.xlsx', index=False)

if __name__ == '__main__':
    print('Ejecución completa')


