import numpy as np
import matplotlib.pyplot as plt
from kalman_ import KalmanFilterReg
import pandas as pd
from correlacion import corr
from datos_yfinance import data
from cointegration import run_adf_test
from cointegration import engle_granger_cointegration_test

# Sectores
comodities = ['CL=F', 'GC=F', 'SI=F']
indices = ["^GSPC", "NDAQ"]
tecnologicas = ["MSFT", "TSLA", "DIS", "GOOGL", "META", 'NVDA', 'PANW', 'TTWO', 'AVGO', 'BRK-B']
software = ['INTC', 'QCOM', 'ORCL', 'IBM']
entretenimiento = ['DIS', 'AMZN', "NFLX", 'WBD', 'CMCSA']
mobilidad = ['GM', 'F', 'TSLA', 'TM', 'NSANY']
armamentos = ['BA', 'OSK', 'RTX', 'BAESY', 'LMT', 'ITA']
construccion = ['TEX']
servicios = ['MELI', 'AMZN']
supermercados = ['WMT', 'COST']
farmaceuticas = ['LLY', 'PFE', 'JNJ']
aerolineas = ['AAL', 'UAL']
petroleras = ['CVX', 'VLO', 'SHEL']
tipo_de_cambio = ['MXN=X', 'eurusd=x', 'gbpusd=x']

tickers = comodities + software + tecnologicas + entretenimiento + mobilidad + armamentos + construccion + servicios + supermercados + farmaceuticas + tipo_de_cambio + aerolineas + petroleras + tipo_de_cambio
star_due = "2015-08-22"

df, tipo_de_cambio = data(tickers, star_due)
df = df.dropna()
df = (df - df.mean()) / df.std()
corr_matrix = df.corr()
cor, actives = corr(corr_matrix)
df = pd.DataFrame(df)

print(corr_matrix)
print(cor, actives)

actives= ['GC=F', 'SI=F']

activo1, activo2 = actives[:2]  # Tomamos los dos activos más correlacionados
series1 = df[activo1]
series2 = df[activo1]

# Ejecutar las pruebas
print(f"\n--- Test de Estacionariedad ADF ---")
p_value_1 = run_adf_test(series1, activo1)
p_value_2 = run_adf_test(series2, activo2)

print(f"\n--- Test de Cointegración (Engle-Granger) ---")
df_test = pd.concat([series1, series2], axis=1).dropna()
p_value_cointegration = engle_granger_cointegration_test(df_test)
if p_value_cointegration < 0.05:
    print(f"Las series están cointegradas (p-valor: {p_value_cointegration:.4f})")
else:
    print(f"No hay evidencia de cointegración (p-valor: {p_value_cointegration:.4f})")

# Graficar los activos
plt.figure(figsize=(12, 6))
plt.plot(series1.index, series1, label=activo1)
plt.plot(series2.index, series2, label=activo2)
plt.legend()
plt.title("Comparación de Activos")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.grid()
plt.show()

if __name__ == '__main__':
    print('PyCharm')

