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


def backtest_estrategia(data, signals, hedge_ratios, capital_inicial=1_000_000, comision=0.00125):
    """
    Backtest de estrategia utilizando:
    - Señales generadas con VECM.
    - Hedge Ratio dinámico obtenido con el filtro de Kalman.
    - Cierre de posiciones cuando el spread regresa a la media.
    - Cálculo del valor del portafolio con múltiples posiciones abiertas.
    - Corrección para evitar caída abrupta del portafolio al abrir posiciones.
    - Corrección del manejo de posiciones SHORT para reflejar correctamente el PnL.
    - Uso del mismo Hedge Ratio en la apertura y el cierre de posiciones.
    - Registro de todos los trades efectuados.
    """

    capital = capital_inicial
    active_positions = []  # Lista de posiciones largas abiertas
    active_short_positions = []  # Lista de posiciones cortas abiertas
    portfolio_value = []  # Almacena el valor del portafolio en cada iteración
    trades_log = []  # Registro de trades ejecutados

    for i, row in data.iterrows():
        signal = signals.loc[row.name, 'signal'] if row.name in signals.index else 0
        hedge_ratio = hedge_ratios.loc[row.name] if row.name in hedge_ratios.index else 1

        # 🔹 Cerrar posiciones cuando el spread regresa a la media
        if abs(signals.loc[row.name, 'ECT'] - signals['ECT'].mean()) < 0.01:
            for position in active_positions:
                pnl = (row['CVX'] - position['bought_at']) * position['shares'] - (
                            position['shares'] * row['CVX'] * comision)
                capital += pnl + (position['shares'] * row['CVX']) * (
                            1 - comision)  # Ajuste para reflejar correctamente PnL
                trades_log.append({
                    'Fecha Entrada': position['date'],
                    'Fecha Cierre': row.name,
                    'Tipo': 'Long',
                    'Activo': 'CVX',
                    'Unidades': position['shares'],
                    'Precio Entrada': position['bought_at'],
                    'Precio Salida': row['CVX'],
                    'Hedge Ratio': position['hedge_ratio'],
                    'PnL': pnl,
                    'Capital': capital
                })
            active_positions = []

            for position in active_short_positions:
                pnl = (position['sell_at'] - row['VLO']) * position['shares'] * position['hedge_ratio'] - (
                            row['VLO'] * position['shares'] * position['hedge_ratio'] * comision)
                capital += pnl  # Ajuste para reflejar correctamente PnL
                trades_log.append({
                    'Fecha Entrada': position['date'],
                    'Fecha Cierre': row.name,
                    'Tipo': 'Short',
                    'Activo': 'VLO',
                    'Unidades': position['shares'],
                    'Precio Entrada': position['sell_at'],
                    'Precio Salida': row['VLO'],
                    'Hedge Ratio': position['hedge_ratio'],
                    'PnL': pnl,
                    'Capital': capital
                })
            active_short_positions = []

        # 🔹 Apertura de nuevas posiciones en pares (LONG en un activo y SHORT en otro simultáneamente)
        if signal == 1:  # Señal de LONG en CVX y SHORT en VLO
            capital_trade = capital * 0.1  # Asignamos el 10% del capital
            units_cvx = capital_trade / row['CVX']  # Unidades de CVX
            units_vlo = units_cvx * abs(hedge_ratio)  # Unidades de VLO usando el hedge ratio

            operation_cost = (units_cvx * row['CVX'] + units_vlo * row['VLO']) * (1 + comision)
            if (capital > operation_cost) and (capital > 250_000):
                capital -= (units_cvx * row['CVX']) * (1 + comision)  # Ajuste para evitar impacto negativo del short

                active_positions.append({
                    'date': row.name,
                    'bought_at': row['CVX'],
                    'shares': units_cvx,
                    'hedge_ratio': hedge_ratio  # Guardamos el hedge ratio en la apertura
                })
                active_short_positions.append({
                    'date': row.name,
                    'sell_at': row['VLO'],
                    'shares': units_vlo,
                    'hedge_ratio': hedge_ratio  # Guardamos el hedge ratio en la apertura
                })

        if signal == -1:  # Señal de SHORT en CVX y LONG en VLO
            capital_trade = capital * 0.1
            units_cvx = capital_trade / row['CVX']
            units_vlo = units_cvx * abs(hedge_ratio)

            operation_cost = (units_cvx * row['CVX'] + units_vlo * row['VLO']) * (1 + comision)
            if (capital > operation_cost) and (capital > 250_000):
                capital -= (units_vlo * row['VLO']) * (1 + comision)  # Ajuste para evitar impacto negativo del short

                active_short_positions.append({
                    'date': row.name,
                    'sell_at': row['CVX'],
                    'shares': units_cvx,
                    'hedge_ratio': hedge_ratio  # Guardamos el hedge ratio en la apertura
                })
                active_positions.append({
                    'date': row.name,
                    'bought_at': row['VLO'],
                    'shares': units_vlo,
                    'hedge_ratio': hedge_ratio  # Guardamos el hedge ratio en la apertura
                })

        # 🔹 Cálculo del Valor del Portafolio considerando correctamente la posición recién abierta
        long_value = sum(
            [pos['shares'] * row['CVX'] for pos in active_positions])  # Valor de mercado de todas las posiciones LONG
        short_pnl = sum([(pos['sell_at'] - row['VLO']) * pos['shares'] * pos['hedge_ratio'] for pos in
                         active_short_positions])  # P&L de todas las posiciones SHORT

        portfolio_value.append({
            'Fecha': row.name,
            'Equity': capital + long_value + short_pnl  # Incluyendo el valor de todas las posiciones abiertas
        })

    trades_df = pd.DataFrame(trades_log)
    equity_df = pd.DataFrame(portfolio_value).set_index('Fecha')

    return equity_df, trades_df


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

    # Marcar los trades cerrados con puntos verdes
    for _, trade in trades_log_df.iterrows():
        plt.scatter(trade["Fecha Cierre"], trade["Capital"], color="green", marker="o", label="Complete Trade" if _ == 0 else "")

    plt.title("Capital Evolution with Complete Trades")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid()
    plt.show()

def graficar_activos_vs_estrategia(precios, equity_curve, trades_log_df, vecm_signals):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # 📊 Precios normalizados
    precios_norm = precios / precios.iloc[0]
    equity_norm = equity_curve / equity_curve.iloc[0]

    ax1.plot(precios_norm.index, precios_norm['CVX'], label='CVX (Normalized)', color='blue')
    ax1.plot(precios_norm.index, precios_norm['VLO'], label='VLO (Normalized)', color='orange')
    ax1.plot(equity_norm.index, equity_norm['Equity'], label='Pairs Trading Strategy', color='black', linewidth=2)

    # 📌 Extraer señales
    short_signals = vecm_signals[vecm_signals['signal'] == -1]
    long_signals = vecm_signals[vecm_signals['signal'] == 1]

    # 🔴 Marcar señales sobre los precios normalizados
    ax1.scatter(short_signals.index, precios_norm['CVX'].reindex(short_signals.index), marker='v', color='red', s=100, label='Short CVX')
    ax1.scatter(short_signals.index, precios_norm['VLO'].reindex(short_signals.index), marker='^', color='green', s=100, label='Long VLO')
    ax1.scatter(long_signals.index, precios_norm['VLO'].reindex(long_signals.index), marker='v', color='red', s=100, label='Short VLO')
    ax1.scatter(long_signals.index, precios_norm['CVX'].reindex(long_signals.index), marker='^', color='green', s=100, label='Long CVX')

    # 🔴 Marcar trades cerrados sobre la curva de capital
    for _, trade in trades_log_df.iterrows():
        ax1.scatter(trade["Fecha Cierre"], equity_norm.loc[trade["Fecha Cierre"], "Equity"],
                    color="red", marker="o", s=50, label="Trade Cerrado" if _ == 0 else "")

    ax1.set_title("Activos vs Pairs Trading")
    ax1.set_ylabel("Normalized Value")
    ax1.legend()
    ax1.grid()

    # 📉 Subgráfico inferior: Spread (ECT) con ±1.5σ y señales
    ax2.plot(vecm_signals.index, vecm_signals['ECT'], label='Spread (ECT)', color='purple')
    ax2.axhline(vecm_signals['ECT'].mean() + 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='+1.5 Sigma')
    ax2.axhline(vecm_signals['ECT'].mean() - 1.5 * vecm_signals['ECT'].std(), color='blue', linestyle='--', label='-1.5 Sigma')
    ax2.axhline(vecm_signals['ECT'].mean(), color='red', linestyle='--', label='ECT Mean')

    ax2.scatter(short_signals.index, short_signals['ECT'], marker='v', color='red', s=100, label='Sell Signal')
    ax2.scatter(long_signals.index, long_signals['ECT'], marker='^', color='green', s=100, label='Buy Signal')

    ax2.set_title("Spread Evolution (ECT) with ±1.5σ and Trading Signals")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("ECT")
    ax2.legend()
    ax2.grid()

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


def backtest_estrategia_control_riesgo(mu, precios, hedge_ratios, capital_inicial=1_000_000, std_threshold=1.5,
                                       comision=0.00125, margen_requerido=0.3):
    """
    Backtest con PnL flotante correctamente reflejado en la curva de capital, siguiendo la estructura:
    Valor del Portafolio = Capital Disponible + Valor de Mercado de Longs + P&L Shorts.
    Se incluye manejo de márgenes para posiciones short.
    """
    señales = mu['signal']
    capital_disponible = capital_inicial  # Capital inicial
    equity_curve = []  # Evolución del capital en cada iteración
    posiciones_abiertas = []
    trades_log = []

    mu_mean = mu['ECT'].mean()  # Media del spread

    for i in range(1, len(mu)):
        fecha = mu.index[i]
        signal = señales.iloc[i]
        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]
        hedge_ratio = hedge_ratios.iloc[i]  # Hedge Ratio dinámico de Kalman

        capital_trade = capital_disponible * 0.1  # Cada trade usa el 10% del capital disponible

        if signal != 0 and capital_disponible >= 0.2 * capital_inicial:
            units_cvx = capital_trade / price_cvx
            units_vlo = units_cvx * abs(hedge_ratio)

            comision_cvx = abs(units_cvx) * price_cvx * comision
            comision_vlo = abs(units_vlo) * price_vlo * comision
            comision_total_apertura = comision_cvx + comision_vlo

            capital_disponible -= (capital_trade + comision_total_apertura)

            nueva_posicion = {
                'Fecha': fecha,
                'Señal': 'Long CVX - Short VLO' if signal == 1 else 'Short CVX - Long VLO',
                'Unidades_CVX': units_cvx if signal == 1 else -units_cvx,
                'Unidades_VLO': -units_vlo if signal == 1 else units_vlo,
                'Precio_entrada_CVX': price_cvx,
                'Precio_entrada_VLO': price_vlo,
                'Hedge Ratio': hedge_ratio,
                'Capital_asignado': capital_trade,
                'Abierta': True
            }
            posiciones_abiertas.append(nueva_posicion)

        # 🔹 Calcular Valor de Mercado de Longs y P&L de Shorts
        valor_mercado_longs = 0
        pnl_shorts = 0

        for posicion in posiciones_abiertas:
            if not posicion['Abierta']:
                continue

            if posicion['Señal'] == 'Long CVX - Short VLO':
                valor_mercado_longs += posicion['Unidades_CVX'] * price_cvx
                pnl_shorts += (posicion['Precio_entrada_VLO'] - price_vlo) * posicion[
                    'Unidades_VLO']  # Short: entrada - precio actual
                margen_retenido = abs(posicion['Unidades_VLO'] * price_vlo) * margen_requerido
                capital_disponible -= margen_retenido  # Reservar margen para posiciones short

            elif posicion['Señal'] == 'Short CVX - Long VLO':
                valor_mercado_longs += posicion['Unidades_VLO'] * price_vlo
                pnl_shorts += (posicion['Precio_entrada_CVX'] - price_cvx) * posicion[
                    'Unidades_CVX']  # Short: entrada - precio actual
                margen_retenido = abs(posicion['Unidades_CVX'] * price_cvx) * margen_requerido
                capital_disponible -= margen_retenido  # Reservar margen para posiciones short

        # 🔹 Calcular el valor del portafolio con la nueva estructura
        valor_portafolio = capital_disponible + valor_mercado_longs + pnl_shorts
        equity_curve.append(valor_portafolio)  # Guardamos la evolución del portafolio

        # 🔴 Cierre de posiciones si el spread regresa a la media
        if abs(mu['ECT'].iloc[i] - mu_mean) < 0.01 and posiciones_abiertas:
            for posicion in posiciones_abiertas:
                if not posicion['Abierta']:
                    continue

                units_cvx_salida = posicion['Unidades_CVX']
                units_vlo_salida = posicion['Unidades_VLO']

                pnl_cvx = (price_cvx - posicion['Precio_entrada_CVX']) * units_cvx_salida
                pnl_vlo = (price_vlo - posicion['Precio_entrada_VLO']) * units_vlo_salida

                comision_cvx_salida = abs(units_cvx_salida) * price_cvx * comision
                comision_vlo_salida = abs(units_vlo_salida) * price_vlo * comision
                comision_total_cierre = comision_cvx_salida + comision_vlo_salida

                pnl_total = pnl_cvx + pnl_vlo - comision_total_cierre
                capital_disponible += pnl_total  # Se suman las ganancias/pérdidas al capital

                posicion.update({
                    'Precio_salida_CVX': price_cvx,
                    'Precio_salida_VLO': price_vlo,
                    'PnL': pnl_total,
                    'Capital': capital_disponible,
                    'Fecha_cierre': fecha,
                    'Abierta': False
                })
                trades_log.append(posicion)

            posiciones_abiertas = []  # Todas las posiciones se cierran

    trades_log_df = pd.DataFrame(trades_log)
    backtest_result = pd.DataFrame(index=mu.index[1:], data={'Equity': equity_curve})

    return backtest_result, trades_log_df