import yfinance as yf

def data_(tickers, start):
    o_general = yf.download(tickers, start=start)["Close"]
    o_general = o_general.dropna()
    tipo_de_cambio = o_general['MXN=X']
    general = o_general
    return o_general, tipo_de_cambio