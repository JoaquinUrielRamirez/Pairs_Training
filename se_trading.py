import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_signals_with_kalman(log_df, threshold_sigma=1.5):

    log_df_shift = log_df.shift(1).dropna()

    # Spread dinámico basado en hedge ratio de Kalman
    ect = log_df_shift['CVX'] - log_df_shift['VLO'] * log_df_shift['hedge_ratio']

    # Calcular umbrales
    ect_mean = ect.mean()
    ect_std = ect.std()
    up_threshold = ect_mean + threshold_sigma * ect_std
    down_threshold = ect_mean - threshold_sigma * ect_std

    # Generar señales
    signals = ect.apply(lambda x: -1 if x > up_threshold else (1 if x < down_threshold else 0))
    signals_df = pd.DataFrame({'ECT': ect, 'signal': signals})

    return signals_df

def generar_senales_trading(mu, std_threshold=1.5):

    mu_mean = mu.mean()
    mu_std = mu.std()
    upper_bound = mu_mean + std_threshold * mu_std
    lower_bound = mu_mean - std_threshold * mu_std

    señales = pd.DataFrame(index=mu.index)
    mu_selected = mu.iloc[:, 0] if isinstance(mu, pd.DataFrame) else mu
    señales['Short_CVX_Long_VLO'] = (mu_selected > upper_bound).astype(int)
    señales['Short_VLO_Long_CVX'] = (mu_selected < lower_bound).astype(int)

    return señales

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

def backtest_estrategia(mu, precios, capital_inicial=1_000_000, std_threshold=1.5, comision=0.00125, target_ganancia=0.25):

    señales = generar_senales_trading(mu, std_threshold)

    capital = capital_inicial
    capital_disponible = capital_inicial  # Nuevo: capital libre para abrir trades
    posicion_por_trade = 0.05
    equity_curve = [capital_inicial]

    posiciones_abiertas = []
    trades_log = []

    for i in range(1, len(mu)):
        fecha = mu.index[i]
        signal_short_cvx_long_vlo = señales['Short_CVX_Long_VLO'].iloc[i]
        signal_short_vlo_long_cvx = señales['Short_VLO_Long_CVX'].iloc[i]
        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]

        # Evaluar si podemos abrir nueva posición
        if (signal_short_cvx_long_vlo or signal_short_vlo_long_cvx) and capital_disponible >= capital * posicion_por_trade:

            capital_trade = capital * posicion_por_trade
            units_cvx = capital_trade / 2 / price_cvx
            units_vlo = capital_trade / 2 / price_vlo

            if signal_short_cvx_long_vlo:
                nueva_posicion = {
                    'Fecha': fecha,
                    'Señal': 'Short CVX - Long VLO',
                    'Unidades_CVX': -units_cvx,
                    'Unidades_VLO': units_vlo,
                    'Precio_entrada_CVX': price_cvx,
                    'Precio_entrada_VLO': price_vlo,
                    'Capital_asignado': capital_trade,
                    'Abierta': True
                }
            elif signal_short_vlo_long_cvx:
                nueva_posicion = {
                    'Fecha': fecha,
                    'Señal': 'Short VLO - Long CVX',
                    'Unidades_CVX': units_cvx,
                    'Unidades_VLO': -units_vlo,
                    'Precio_entrada_CVX': price_cvx,
                    'Precio_entrada_VLO': price_vlo,
                    'Capital_asignado': capital_trade,
                    'Abierta': True
                }

            posiciones_abiertas.append(nueva_posicion)
            capital_disponible -= capital_trade  # Descontamos lo usado en este trade

        # Revisar si alguna posición debe cerrarse
        for posicion in posiciones_abiertas:
            if not posicion['Abierta']:
                continue

            pnl_cvx = pnl_vlo = 0

            if posicion['Unidades_CVX'] > 0:  # Long CVX
                pnl_cvx = posicion['Unidades_CVX'] * (price_cvx - posicion['Precio_entrada_CVX'])
            elif posicion['Unidades_CVX'] < 0:  # Short CVX
                pnl_cvx = abs(posicion['Unidades_CVX']) * (posicion['Precio_entrada_CVX'] - price_cvx)

            if posicion['Unidades_VLO'] > 0:  # Long VLO
                pnl_vlo = posicion['Unidades_VLO'] * (price_vlo - posicion['Precio_entrada_VLO'])
            elif posicion['Unidades_VLO'] < 0:  # Short VLO
                pnl_vlo = abs(posicion['Unidades_VLO']) * (posicion['Precio_entrada_VLO'] - price_vlo)

            pnl_total = pnl_cvx + pnl_vlo

            comision_cvx = abs(posicion['Unidades_CVX']) * (posicion['Precio_entrada_CVX'] + price_cvx) * 0.5 * comision
            comision_vlo = abs(posicion['Unidades_VLO']) * (posicion['Precio_entrada_VLO'] + price_vlo) * 0.5 * comision
            costo_operacion = comision_cvx + comision_vlo

            pnl_neto = pnl_total - costo_operacion

            if pnl_neto >= target_ganancia * posicion['Capital_asignado']:
                capital += pnl_neto
                capital_disponible += posicion['Capital_asignado'] + pnl_neto  # Regresamos lo invertido + ganancia

                posicion.update({
                    'Precio_salida_CVX': price_cvx,
                    'Precio_salida_VLO': price_vlo,
                    'PnL': pnl_neto,
                    'Capital': capital,
                    'Fecha_cierre': fecha,
                    'Abierta': False
                })

                trades_log.append(posicion)

        equity_curve.append(capital)

    trades_log_df = pd.DataFrame(trades_log)
    backtest_result = pd.DataFrame(index=mu.index, data={'Equity': equity_curve})

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

