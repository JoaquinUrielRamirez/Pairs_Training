import numpy as np
import matplotlib.pyplot as plt
from kalman_ import KalmanFilterReg
import pandas as pd
from correlacion import corr
from datos_yfinance import data
from cointegration import adf_test
from cointegration import regression_cointegration
from cointegration import johansen_test
from cointegration import test_cointegration

# Sectores
comodities = ['CL=F', 'GC=F', 'SI=F']
indices = ["^GSPC", "NDAQ"]
tecnologicas = ["MSFT", "TSLA", "DIS", "GOOGL", "META", 'NVDA', 'PANW', 'TTWO', 'AVGO']
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
tipo_de_cambio = ['MXN=X']

tickers = comodities + tecnologicas + software + entretenimiento + mobilidad + armamentos + construccion + servicios + supermercados + farmaceuticas + tipo_de_cambio + aerolineas + petroleras
star_due = "2015-08-22"

df, tipo_de_cambio = data(tickers, star_due)
corr_matrix = df.corr()
cor, actives = corr(corr_matrix)

print(cor, actives)

#print(adf_test(data))

if __name__ == '__main__':
    print('PyCharm')

