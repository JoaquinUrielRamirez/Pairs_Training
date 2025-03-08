import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def graficar_estrategia_y_ect(norm, vecm_signals):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # 📊 Precios normalizados
    ax1.plot(norm.index, norm['CVX'], label='CVX', color='blue')
    ax1.plot(norm.index, norm['VLO'], label='VLO', color='black')

    # 📌 Extraer señales
    short_signals = vecm_signals[vecm_signals['signal'] == -1]
    long_signals = vecm_signals[vecm_signals['signal'] == 1]

    # 🔴 Marcar señales sobre los precios normalizados
    ax1.scatter(short_signals.index, norm['CVX'].reindex(short_signals.index), marker='v', color='red', s=100, label='Short CVX')
    ax1.scatter(short_signals.index, norm['VLO'].reindex(short_signals.index), marker='^', color='green', s=100, label='Long VLO')
    ax1.scatter(long_signals.index, norm['VLO'].reindex(long_signals.index), marker='v', color='red', s=100, label='Short VLO')
    ax1.scatter(long_signals.index, norm['CVX'].reindex(long_signals.index), marker='^', color='green', s=100, label='Long CVX')

    ax1.set_title("CVX vs VLO (10 years) + Pairs Trading Signals")
    ax1.set_ylabel("Normalized Price")
    ax1.grid(True)
    ax1.legend()

    # 📈 Graficar ECT y señales
    ax2.plot(vecm_signals.index, vecm_signals['ECT'], label='ECT', color='purple')
    ax2.axhline(vecm_signals['ECT'].mean() + 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='+1.5 Sigma')
    ax2.axhline(vecm_signals['ECT'].mean() - 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='-1.5 Sigma')
    ax2.axhline(vecm_signals['ECT'].mean(), color='red', linestyle='--', label='ECT Mean')

    ax2.scatter(short_signals.index, short_signals['ECT'], marker='v', color='red', s=100, label='Sell Signal')
    ax2.scatter(long_signals.index, long_signals['ECT'], marker='^', color='green', s=100, label='Buy Signal')

    ax2.set_title("VECM: ECT and Trading Signals (±1.5σ)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("ECT")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def backtest_estrategia(mu, precios, hedge_ratios, capital_inicial=1_000_000, std_threshold=1.5, comision=0.00125):
    """
    Backtest de una estrategia de pairs trading con hedge ratio dinámico basado en Kalman Filter,
    asegurando que TODAS las posiciones abiertas se cierren cuando el spread regresa a su media.
    """
    señales = mu['signal']
    capital = capital_inicial
    posicion_por_trade = 0.1  # Cada trade usa 10% del capital disponible
    equity_curve = []  # Evolución del capital en cada iteración
    posiciones_abiertas = []
    trades_log = []

    mu_mean = mu['ECT'].mean()  # Asegurar que tomamos solo la columna ECT y convertimos a escalar

    for i in range(1, len(mu)):
        fecha = mu.index[i]
        signal = señales.iloc[i]
        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]
        hedge_ratio = hedge_ratios.iloc[i]  # Hedge Ratio dinámico de Kalman

        capital_trade = capital * posicion_por_trade

        if signal != 0:
            units_cvx = capital_trade / (1 + abs(hedge_ratio)) / price_cvx
            units_vlo = units_cvx * hedge_ratio

            nueva_posicion = {
                'Fecha': fecha,
                'Señal': 'Long CVX - Short VLO' if signal == 1 else 'Short CVX - Long VLO',
                'Unidades_CVX': units_cvx if signal == 1 else -units_cvx,
                'Unidades_VLO': -units_vlo if signal == 1 else units_vlo,
                'Precio_entrada_CVX': price_cvx,
                'Precio_entrada_VLO': price_vlo,
                'Hedge Ratio': hedge_ratio,
                'Capital_asignado': capital_trade if signal == 1 else 0,  # Capital solo se usa en LONG
                'Abierta': True
            }
            posiciones_abiertas.append(nueva_posicion)

        # Evaluar la evolución del capital con posiciones abiertas
        capital_temp = capital
        for posicion in posiciones_abiertas:
            if not posicion['Abierta']:
                continue

            pnl_cvx = (price_cvx - posicion['Precio_entrada_CVX']) * posicion['Unidades_CVX']
            pnl_vlo = (price_vlo - posicion['Precio_entrada_VLO']) * posicion['Unidades_VLO']

            capital_temp += pnl_cvx + pnl_vlo  # Capital solo cambia en LONG o cuando cierra un SHORT

        equity_curve.append(capital_temp)

        # Cierre de TODAS las posiciones si el spread regresa a la media
        if abs(mu['ECT'].iloc[i] - mu_mean) < 0.01 and posiciones_abiertas:
            for posicion in posiciones_abiertas:
                if not posicion['Abierta']:
                    continue

                hedge_ratio_original = posicion['Hedge Ratio']
                units_cvx_salida = posicion['Unidades_CVX']
                units_vlo_salida = posicion['Unidades_VLO']

                pnl_cvx = (price_cvx - posicion['Precio_entrada_CVX']) * units_cvx_salida
                pnl_vlo = (price_vlo - posicion['Precio_entrada_VLO']) * units_vlo_salida

                comision_cvx = abs(units_cvx_salida) * (posicion['Precio_entrada_CVX'] + price_cvx) * 0.5 * comision
                comision_vlo = abs(units_vlo_salida) * (posicion['Precio_entrada_VLO'] + price_vlo) * 0.5 * comision

                pnl_total = pnl_cvx + pnl_vlo - (comision_cvx + comision_vlo)

                if posicion['Señal'] == 'Short CVX - Long VLO':
                    capital += pnl_total  # El capital solo cambia al cerrar un SHORT

                posicion.update({
                    'Precio_salida_CVX': price_cvx,
                    'Precio_salida_VLO': price_vlo,
                    'PnL': pnl_total,
                    'Capital': capital,
                    'Fecha_cierre': fecha,
                    'Abierta': False
                })
                trades_log.append(posicion)

            # Una vez cerradas todas, vaciar la lista de posiciones abiertas
            posiciones_abiertas = []

    trades_log_df = pd.DataFrame(trades_log)
    backtest_result = pd.DataFrame(index=mu.index[1:], data={'Equity': equity_curve})

    return backtest_result, trades_log_df


def calcular_metrica_backtest(trades_log_df, equity_curve):

    total_trades = len(trades_log_df)
    trades_ganadores = trades_log_df[trades_log_df["PnL"] > 0]
    porcentaje_ganador = len(trades_ganadores) / total_trades * 100 if total_trades > 0 else 0

    # Asegurar que equity_array sea un vector 1D
    equity_array = np.array(equity_curve["Equity"]).flatten()

    if len(equity_array) < 2:
        max_drawdown = np.nan
    else:
        # Drawdown máximo
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - max_equity) / max_equity
        max_drawdown = np.min(drawdown)

    # Crear DataFrame con métricas
    metricas_df = pd.DataFrame({
        "Total Trades": [total_trades],
        "% Trades Ganadores": [porcentaje_ganador],
        "Drawdown Máximo": [max_drawdown],
    })

    return metricas_df

def graficar_equity_curve(equity_curve, trades_log_df):

    plt.figure(figsize=(12,6))
    plt.plot(equity_curve.index, equity_curve["Equity"], label="Capital Evolution", color="royalblue")

    # Marcar los trades cerrados con puntos rojos
    for _, trade in trades_log_df.iterrows():
        plt.scatter(trade["Fecha_cierre"], trade["Capital"], color="green", marker="o", label="Complete Trade" if _ == 0 else "")

    plt.title("Capital Evolution with Complete Trades")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid()
    plt.show()

def graficar_activos_vs_estrategia(precios, equity_curve, trades_log_df, mu):

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # Dos subgráficos: 2/3 arriba y 1/3 abajo

    # 📈 Subgráfico superior: Activos y estrategia
    ax1 = plt.subplot(gs[0])

    # Normalizar precios y capital
    precios_norm = precios / precios.iloc[0]
    equity_norm = equity_curve / equity_curve.iloc[0]

    ax1.plot(precios_norm.index, precios_norm["CVX"], label="CVX (Normalized)", color="blue", alpha=0.6)
    ax1.plot(precios_norm.index, precios_norm["VLO"], label="VLO (Normalized)", color="black", alpha=0.6)
    ax1.plot(equity_norm.index, equity_norm["Equity"], label="Pairs Trading Strategy", color="green", linewidth=2)

    # 🔴 Marcar trades cerrados sobre la curva de capital
    for _, trade in trades_log_df.iterrows():
        ax1.scatter(trade["Fecha_cierre"], equity_norm.loc[trade["Fecha_cierre"], "Equity"],
                    color="red", marker="o", s=50, label="Trade Cerrado" if _ == 0 else "")

    ax1.set_title("Actives vs Pairs Trading")
    ax1.set_ylabel("Normalized Value")
    ax1.legend()
    ax1.grid()

    # 📉 Subgráfico inferior: Spread (mu) con ±1.5σ y señales
    ax2 = plt.subplot(gs[1], sharex=ax1)  # Comparte el eje X con la gráfica superior

    # ✅ Media y desviación estándar constantes
    mu_mean = mu.mean()
    mu_std = mu.std()

    # ✅ Expandir ±1.5σ para que tengan el mismo índice que mu
    upper_band = pd.Series(mu_mean + 1.5 * mu_std, index=mu.index)
    lower_band = pd.Series(mu_mean - 1.5 * mu_std, index=mu.index)

    # 🔹 Graficar el spread con color fuerte y mayor grosor
    ax2.plot(mu.index, mu, label="Spread (ECT)", color="purple", linestyle="solid", linewidth=1.5)

    # 🔹 Graficar bandas de ±1.5σ y la media
    ax2.plot(mu.index, upper_band, label="+1.5 Sigma", color="blue", linestyle="dashed", linewidth=1.2)
    ax2.plot(mu.index, lower_band, label="-1.5 Sigma", color="blue", linestyle="dashed", linewidth=1.2)
    ax2.plot(mu.index, [mu_mean] * len(mu), label="ECT Mean", color="red", linestyle="dashed", linewidth=1.2)

    # 🔴 Señales de Venta (cuando el spread cruza arriba de +1.5σ)
    ventas = mu > upper_band
    ax2.scatter(mu.index[ventas], mu[ventas], color="red", marker="v", s=40, label="Sell Signal")

    # 🟢 Señales de Compra (cuando el spread cruza abajo de -1.5σ)
    compras = mu < lower_band
    ax2.scatter(mu.index[compras], mu[compras], color="green", marker="^", s=40, label="Buy Signal")

    ax2.set_title("Spread Evolution (ECT) with  ±1.5σ and Trading Signals")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("ECT")
    ax2.legend()
    ax2.grid()

    # Ajustar los subgráficos
    plt.tight_layout()
    plt.show()

def graficar_spread_trading(mu):

    fig, ax = plt.subplots(figsize=(14, 5))

    # ✅ Calcular media y desviación estándar constantes
    mu_mean = mu.mean()
    mu_std = mu.std()

    # ✅ Crear bandas ±1.5σ con el mismo índice que mu
    upper_band = pd.Series(mu_mean + 1.5 * mu_std, index=mu.index)
    lower_band = pd.Series(mu_mean - 1.5 * mu_std, index=mu.index)

    # 🔹 Graficar el spread con color fuerte y mayor grosor
    ax.plot(mu.index, mu, label="ECT", color="purple", linestyle="solid", linewidth=1.5)

    # 🔹 Graficar bandas de ±1.5σ y la media
    ax.plot(mu.index, upper_band, label="+1.5 Sigma", color="blue", linestyle="dashed", linewidth=1.2)
    ax.plot(mu.index, lower_band, label="-1.5 Sigma", color="blue", linestyle="dashed", linewidth=1.2)
    ax.plot(mu.index, [mu_mean] * len(mu), label="ECT Mean", color="red", linestyle="dashed", linewidth=1.2)

    # 🔴 Señales de Venta (cuando el spread cruza arriba de +1.5σ)
    ventas = mu > upper_band
    ax.scatter(mu.index[ventas], mu[ventas], color="red", marker="v", s=40, label="Sell Signal")

    # 🟢 Señales de Compra (cuando el spread cruza abajo de -1.5σ)
    compras = mu < lower_band
    ax.scatter(mu.index[compras], mu[compras], color="green", marker="^", s=40, label="Buy Signal")

    # 📌 Configuración de la gráfica
    ax.set_title("VECM: ECT and Trading Signals (±1.5σ)")
    ax.set_xlabel("Date")
    ax.set_ylabel("ECT")
    ax.legend()
    ax.grid()
    plt.show()