from statsmodels.tsa.stattools import adfuller, coint

def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"\n{name}:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("La serie es estacionaria (rechaza H0).")
    else:
        print("La serie NO es estacionaria (no rechaza H0).")