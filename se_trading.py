import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generar_senales_trading(mu, std_threshold=1.5):
    """
    Genera señales de trading basadas en el término de corrección de error.
    """
    mu_mean = mu.mean()
    mu_std = mu.std()
    upper_bound = mu_mean + std_threshold * mu_std
    lower_bound = mu_mean - std_threshold * mu_std

    señales = pd.DataFrame(index=mu.index)
    mu_selected = mu.iloc[:, 0] if isinstance(mu, pd.DataFrame) else mu
    señales['Short_CVX_Long_VLO'] = (mu_selected > upper_bound).astype(int)
    señales['Short_VLO_Long_CVX'] = (mu_selected < lower_bound).astype(int)

    return señales


def backtest_estrategia(mu, precios, capital_inicial=1_000_000, std_threshold=1.5, comision=0.00125):
    """
    Realiza un backtest considerando apertura y cierre de posiciones.
    """
    señales = generar_senales_trading(mu, std_threshold)
    capital = capital_inicial
    posicion_por_trade = 0.1  # 10% del capital en cada trade
    posiciones = {'CVX': 0, 'VLO': 0}  # Seguimiento de posiciones abiertas
    equity_curve = [capital]

    for i in range(1, len(mu)):
        signal_short_cvx_long_vlo = señales['Short_CVX_Long_VLO'].iloc[i]
        signal_short_vlo_long_cvx = señales['Short_VLO_Long_CVX'].iloc[i]
        ect_actual = mu.iloc[i]
        ect_media = mu.mean()

        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]
        capital_trade = capital * posicion_por_trade
        units_cvx = capital_trade / price_cvx
        units_vlo = capital_trade / price_vlo

        # Apertura de posición
        if signal_short_cvx_long_vlo:
            posiciones['CVX'] -= units_cvx  # Short en CVX
            posiciones['VLO'] += units_vlo  # Long en VLO
        elif signal_short_vlo_long_cvx:
            posiciones['CVX'] += units_cvx  # Long en CVX
            posiciones['VLO'] -= units_vlo  # Short en VLO

        # Cierre de posición si el ECT vuelve a la media
        if abs(ect_actual - ect_media) < 0.1:
            pnl_cvx = posiciones['CVX'] * (price_cvx - precios['CVX'].iloc[i - 1])
            pnl_vlo = posiciones['VLO'] * (price_vlo - precios['VLO'].iloc[i - 1])
            costo_operacion = (abs(posiciones['CVX']) * price_cvx + abs(posiciones['VLO']) * price_vlo) * comision
            capital += pnl_cvx + pnl_vlo - costo_operacion
            posiciones = {'CVX': 0, 'VLO': 0}  # Cerrar posiciones

        equity_curve.append(capital)

    backtest_result = pd.DataFrame(index=mu.index, data={'Equity': equity_curve})
    return backtest_result


def visualizar_backtest(backtest_result):
    """
    Grafica la evolución del capital durante el backtest.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_result.index, backtest_result['Equity'], label="Capital", color='blue')
    plt.axhline(y=backtest_result['Equity'].iloc[0], color='black', linestyle='--', label="Capital Inicial")
    plt.title("Backtest de la Estrategia de Pairs Trading")
    plt.xlabel("Fecha")
    plt.ylabel("Capital (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
