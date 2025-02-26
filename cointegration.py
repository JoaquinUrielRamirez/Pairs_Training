import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def run_adf_test(series, name: str):
    series = series.dropna()
    result = adfuller(series)
    print(f"{name} ADF Statistic: {result[0]:.4f}")
    print(f"{name} p-value: {result[1]:.4f}")
    return result[1]  # Devuelve el p-valor

# Verificamos la cointegración con Engle-Granger
def engle_granger_cointegration_test(df):
    series1, series2 = df.iloc[:, 0], df.iloc[:, 1]
    df_reg = sm.add_constant(df)
    model = sm.OLS(df_reg.iloc[:, 0], df_reg.iloc[:, [1, 2]]).fit()
    residuals = model.resid
    p_value = adfuller(residuals.dropna())[1]
    return p_value