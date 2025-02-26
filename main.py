import numpy as np
import matplotlib.pyplot as plt
#from kalman_ import KalmanFilterReg
import pandas as pd
import yfinance as yf
from se_trading import generar_senales_trading, ajustar_estrategia_balanceada, visualizar_senales
from cointegration import analizar_cointegracion, johansen_cointegration_test

tickers = ['CVX', 'VLO']
start_date = "2015-08-22"

data = yf.download(tickers, start=start_date)["Open"]
data = pd.DataFrame(data)
data = data.dropna()
norm = (data - data.mean()) / data.std()

# ✅ 2. Verificar cointegración y correlación
resultados = analizar_cointegracion(data, tickers)
print(resultados)

cor = data.corr()
print(f"Correlación entre {tickers[0]} y {tickers[1]}:", cor[tickers[0]][tickers[1]])
c_johansen = johansen_cointegration_test(data)
print(c_johansen)

spread_jo = 0.08684451 * data['CVX'] - 0.08036587 * data['VLO']
spread_normalizado = (spread_jo - spread_jo.mean()) / spread_jo.std()

# Definir Umbrales +- 1.5 std
threshold_up = 1.5 * spread_jo.std()
threshold_down = -1.5 * spread_jo.std()

#SEÑALES
spread_df = spread_normalizado.to_frame(name='spread').sort_index()
short_cvx_long_vlo = spread_df[spread_df['spread'] > threshold_up]
short_vlo_long_cvx = spread_df[spread_df['spread'] < threshold_down]

# GRAFIC
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,8))
ax1.plot(norm.index, norm['CVX'], label='CVX', color='blue')
ax1.plot(norm.index, norm['VLO'], label='VLO', color='Yellow')

ax1.scatter(short_cvx_long_vlo.index, norm['CVX'].reindex(short_cvx_long_vlo.index), marker='v', color='red', s=100, label='Short CVX')
ax1.scatter(short_cvx_long_vlo.index, norm['VLO'].reindex(short_cvx_long_vlo.index), marker='^', color='green', s=100, label='Long VLO')

ax1.scatter(short_vlo_long_cvx.index, norm['VLO'].reindex(short_vlo_long_cvx.index), marker='v', color='red', s=100, label='Short VLO')
ax1.scatter(short_vlo_long_cvx.index, norm['CVX'].reindex(short_vlo_long_cvx.index), marker='^', color='green', s=100, label='Long CVX')

ax1.set_title("Comparación normalizada (Min-Max) SHEL vs VLO (10 años) + Señales Pairs Trading")
ax1.set_ylabel("Precio Normalizado")
ax1.grid(True)

handles1, labels1 = ax1.get_legend_handles_labels()
# Filtramos duplicados conservando el orden
unique = list(dict(zip(labels1, handles1)).items())
ax1.legend([u[1] for u in unique], [u[0] for u in unique], loc='best')

ax2.plot(spread_df.index, spread_df['spread'], label="Spread (Johansen) centrado", color='magenta')
ax2.axhline(threshold_up, color='blue', linestyle='--', label='+1.5 Sigma')
ax2.axhline(threshold_down, color='blue', linestyle='--', label='-1.5 Sigma')
ax2.axhline(0, color='black', linestyle='--', label='Media 0')
ax2.set_title("Spread (Johansen) con ±1.5 STD")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Spread")
ax2.grid(True)
ax2.legend(loc="best")

plt.tight_layout()
plt.show()










if __name__ == '__main__':
    print('PyCharm')
