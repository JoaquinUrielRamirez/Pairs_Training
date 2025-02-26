import numpy as np
import matplotlib.pyplot as plt
from kalman_ import KalmanFilterReg
import pandas as pd
import yfinance as yf
from stationarity import check_stationarity
from cointegration import analizar_cointegracion

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
tipo_de_cambio = ['MXN=X', 'EURUSD=X', 'GBPUSD=X']

tickers = comodities + indices + software + tecnologicas + entretenimiento + mobilidad + armamentos + construccion + servicios + supermercados + farmaceuticas + aerolineas + petroleras + tipo_de_cambio
star_due = "2015-08-22"

data = yf.download(tickers, start=star_due)["Open"]
data = pd.DataFrame(data)
data = (data - data.mean()) / data.std()

activos = ['AMZN', 'MELI']
resultados = analizar_cointegracion(data, activos)
print(resultados)

cor = data.corr()
print(cor['AMZN']['MELI'])

plt.figure()
plt.grid()
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("Evolución de los Activos")
plt.plot(data.index, data[activos[0]], label=activos[0])
plt.plot(data.index, data[activos[1]], label=activos[1])
plt.legend()
plt.show()

if __name__ == '__main__':
    print('PyCharm')

