import numpy as np
import matplotlib.pyplot as plt
#from kalman_ import KalmanFilterReg
import pandas as pd
import yfinance as yf
from se_trading import generar_senales_trading, ajustar_estrategia_balanceada, visualizar_senales
from cointegration import analizar_cointegracion, johansen_cointegration_test, generate_vecm_signals, run_kalman_filter_custom, KalmanFilterReg, ols_regression_and_plot

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


## Hedge Ratio, Kalman Filter
# Conertir a Logaritmo para más precisión
log_data = np.log(data)
# Definimos: x = log(CVX) y y = log(VLO)
log_x = log_data['CVX']
log_y = log_data['VLO']

kalman_results = run_kalman_filter_custom(log_x, log_y)
# kalman_results contiene 'alpha', 'beta' y 'spread' (dinámico spread = y - (alpha + beta*x))

# 4. Calcular umbrales basados en el spread dinámico (centrado)
spread_dyn = kalman_results['spread']
spread_mean = spread_dyn.mean()
spread_std = spread_dyn.std()
threshold_up = spread_mean + 1.5 * spread_std
threshold_down = spread_mean - 1.5 * spread_std

# Generar señales:
# Si spread > threshold_up: señal -1 (vender spread: short CVX, long VLO)
# Si spread < threshold_down: señal 1 (comprar spread: long CVX, short VLO)
signals = spread_dyn.apply(lambda s: -1 if s > threshold_up else (1 if s < threshold_down else 0))
kalman_results['signal'] = signals

# 5. Graficar la evolución dinámica del hedge ratio (beta)
plt.figure(figsize=(12, 6))
plt.plot(kalman_results.index, kalman_results['beta'], label="Dynamic Beta (Kalman)", color='blue')
plt.title("Evolución Dinámica del Hedge Ratio (Beta) - Kalman Filter")
plt.xlabel("Fecha")
plt.ylabel("Beta")
plt.legend()
plt.grid(True)
plt.show()

# 6. Graficar el spread dinámico con umbrales y marcar señales
plt.figure(figsize=(12, 6))
plt.plot(spread_dyn.index, spread_dyn, label="Dynamic Spread", color='magenta')
plt.axhline(threshold_up, color='blue', linestyle='--', label='Upper Threshold')
plt.axhline(threshold_down, color='blue', linestyle='--', label='Lower Threshold')
plt.axhline(spread_mean, color='red', linestyle='--', label='Spread Mean')

# Marcar las señales:
short_signals = kalman_results[kalman_results['signal'] == -1]
long_signals = kalman_results[kalman_results['signal'] == 1]
plt.scatter(short_signals.index, spread_dyn.loc[short_signals.index], marker='v', color='red', s=100,
            label='Sell Signal')
plt.scatter(long_signals.index, spread_dyn.loc[long_signals.index], marker='^', color='green', s=100,
            label='Buy Signal')

plt.title("Dynamic Spread (Kalman Filter) con Señales ±1.5 STD")
plt.xlabel("Fecha")
plt.ylabel("Spread")
plt.legend()
plt.grid(True)
plt.show()

# Generate Trade Signal susing VECM
# 3. Generar señales de trading usando VECM
vecm_signals, vecm_res = generate_vecm_signals(log_data, det_order=0, k_ar_diff=1, threshold_sigma=1.5)
print("\nSeñales (primeras 10 filas):")
print(vecm_signals.head(10))

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

# 4. Graficar el ECT y las señales
ax2.plot(vecm_signals.index, vecm_signals['ECT'], label='ECT', color='purple')

# Calcular umbrales para graficar
ect_mean = vecm_signals['ECT'].mean()
ect_std = vecm_signals['ECT'].std()
up_line = ect_mean + 1.5 * ect_std
down_line = ect_mean - 1.5 * ect_std

ax2.axhline(up_line, color='blue', linestyle='--', label='+1.5 Sigma')
ax2.axhline(down_line, color='blue', linestyle='--', label='-1.5 Sigma')
ax2.axhline(ect_mean, color='red', linestyle='--', label='ECT Mean')

# Marcar señales
short_signals = vecm_signals[vecm_signals['signal'] == -1]
long_signals = vecm_signals[vecm_signals['signal'] == 1]
ax2.scatter(short_signals.index, short_signals['ECT'], marker='v', color='red', s=100, label='Señal Venta')
ax2.scatter(long_signals.index, long_signals['ECT'], marker='^', color='green', s=100, label='Señal Compra')

plt.title("VECM: ECT y Señales de Trading (±1.5σ)")
plt.xlabel("Fecha")
plt.ylabel("ECT")
plt.legend()
plt.grid(True)
plt.show()

if __name__ == '__main__':
    print('PyCharm')
