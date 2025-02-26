import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def adf_test(data):
    print("Prueba ADF para cada acción:")
    for stock in data.columns:
        result = adfuller(data[stock])
        print(f"{stock}: p-valor = {result[1]}")

def regression_cointegration(data):
    y = data.iloc[:, 0]  # Primera acción como dependiente
    X = data.iloc[:, 1:]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    adf_resid = adfuller(residuals)
    print(f"\nPrueba ADF en los residuales: p-valor = {adf_resid[1]}")

def johansen_test(data):
    johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
    print("\nPrueba de Johansen:")
    print("Estadísticos de Trace:", johansen_result.lr1)
    print("Valores críticos:", johansen_result.cvt)

def test_cointegration(tickers, data):
    adf_test(data)
    regression_cointegration(data)
    if len(tickers) > 2:
        johansen_test(data)
    return data