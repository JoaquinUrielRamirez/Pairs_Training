import numpy as np
import matplotlib.pyplot as plt
from kalman_ import KalmanFilterReg
import pandas as pd
import yfinance as yf
from se_trading import generar_senales_trading, ajustar_estrategia_balanceada, visualizar_senales
from cointegration import analizar_cointegracion, prueba_cointegracion_johansen, entrenar_vecm, aplicar_filtro_kalman_reg, backtest_estrategia_balanceada, visualizar_trading_signals

tickers = ['CVX', 'VLO']
start_date = "2015-08-22"

data = yf.download(tickers, start=start_date)["Open"]
data = pd.DataFrame(data)
data = data.dropna()

# ✅ 2. Verificar cointegración y correlación
resultados = analizar_cointegracion(data, tickers)
print(resultados)

cor = data.corr()
print(f"Correlación entre {tickers[0]} y {tickers[1]}:", cor[tickers[0]][tickers[1]])
c_johansen = prueba_cointegracion_johansen(data, tickers)
print(c_johansen)

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

if __name__ == '__main__':
    print('PyCharm')
