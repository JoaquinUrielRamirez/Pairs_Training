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
    Backtest de una estrategia de pairs trading usando señales de cointegración.
    Incluye log de trades y curva de capital correctamente inicializada.
    """

    # 1️⃣ Generamos señales
    señales = generar_senales_trading(mu, std_threshold)

    # 2️⃣ Inicializamos variables
    capital = capital_inicial
    posicion_por_trade = 0.1  # Cada trade arriesga 10% del capital
    posiciones = {'CVX': 0, 'VLO': 0}
    equity_curve = [capital_inicial]  # <- ARRANCA EXACTO EN CAPITAL INICIAL

    # 3️⃣ Log de trades
    trades_log = []
    precio_entrada_cvx = None
    precio_entrada_vlo = None

    # 4️⃣ Loop del backtest (barremos cada día)
    for i in range(1, len(mu)):
        fecha = mu.index[i]
        signal_short_cvx_long_vlo = señales['Short_CVX_Long_VLO'].iloc[i]
        signal_short_vlo_long_cvx = señales['Short_VLO_Long_CVX'].iloc[i]
        ect_actual = mu.iloc[i]
        ect_media = mu.mean()

        price_cvx = precios['CVX'].iloc[i]
        price_vlo = precios['VLO'].iloc[i]
        capital_trade = capital * posicion_por_trade
        units_cvx = capital_trade / price_cvx
        units_vlo = capital_trade / price_vlo

        # Apertura de posición (si hay señal)
        if signal_short_cvx_long_vlo:
            posiciones['CVX'] -= units_cvx
            posiciones['VLO'] += units_vlo
            precio_entrada_cvx = price_cvx
            precio_entrada_vlo = price_vlo

            trades_log.append({
                'Fecha': fecha,
                'Señal': 'Short CVX - Long VLO',
                'Unidades_CVX': -units_cvx,
                'Unidades_VLO': units_vlo,
                'Precio_entrada_CVX': price_cvx,
                'Precio_entrada_VLO': price_vlo,
                'Precio_salida_CVX': None,
                'Precio_salida_VLO': None,
                'PnL': None,
                'Capital': capital
            })

        elif signal_short_vlo_long_cvx:
            posiciones['CVX'] += units_cvx
            posiciones['VLO'] -= units_vlo
            precio_entrada_cvx = price_cvx
            precio_entrada_vlo = price_vlo

            trades_log.append({
                'Fecha': fecha,
                'Señal': 'Short VLO - Long CVX',
                'Unidades_CVX': units_cvx,
                'Unidades_VLO': -units_vlo,
                'Precio_entrada_CVX': price_cvx,
                'Precio_entrada_VLO': price_vlo,
                'Precio_salida_CVX': None,
                'Precio_salida_VLO': None,
                'PnL': None,
                'Capital': capital
            })

        # Cierre de posición (cuando el spread regresa a la media)
        if abs(ect_actual - ect_media) < 0.1 and (posiciones['CVX'] != 0 or posiciones['VLO'] != 0):
            pnl_cvx = posiciones['CVX'] * (price_cvx - precio_entrada_cvx)
            pnl_vlo = posiciones['VLO'] * (price_vlo - precio_entrada_vlo)
            costo_operacion = (abs(posiciones['CVX']) * price_cvx + abs(posiciones['VLO']) * price_vlo) * comision
            pnl_total = pnl_cvx + pnl_vlo - costo_operacion
            capital += pnl_total

            # Cerrar posiciones
            posiciones = {'CVX': 0, 'VLO': 0}

            # Registrar precios de salida y PnL en el último trade registrado
            trades_log[-1].update({
                'Precio_salida_CVX': price_cvx,
                'Precio_salida_VLO': price_vlo,
                'PnL': pnl_total,
                'Capital': capital
            })

        # Registrar capital al cierre de cada día
        equity_curve.append(capital)

    # 5️⃣ Convertir log de trades a DataFrame
    trades_log_df = pd.DataFrame(trades_log)

    # 6️⃣ Resultado final: equity curve bien armada
    backtest_result = pd.DataFrame(index=mu.index, data={'Equity': equity_curve})

    return backtest_result, trades_log_df


def visualizar_backtest(backtest_result, trades_log, capital_inicial=1_000_000):
    # Calcular PnL acumulado a partir de los trades registrados
    trades_log['PnL_acumulado'] = trades_log['PnL'].cumsum()

    # Crear la figura
    plt.figure(figsize=(14, 7))

    # Línea de evolución de capital (Equity Curve)
    plt.plot(backtest_result.index, backtest_result['Equity'], label='Evolución del Capital', color='royalblue')

    # Línea de revenue acumulado (PnL Acumulado)
    plt.plot(trades_log['Fecha'], capital_inicial + trades_log['PnL_acumulado'],
             label='Capital Inicial + PnL Acumulado', linestyle='--', color='darkorange')

    # Añadir markers para cada trade
    for i, row in trades_log.iterrows():
        fecha = row['Fecha']
        capital = row['Capital']
        if row['Señal'] == 'Short CVX - Long VLO':
            color = 'red'
            marker = 'v'
        else:
            color = 'green'
            marker = '^'
        plt.scatter(fecha, capital, color=color, marker=marker, zorder=5, label=f"Trade {i + 1}" if i < 2 else "")

    # Configuración final de la gráfica
    plt.legend()
    plt.title('Evolución del Capital y PnL Acumulado')
    plt.xlabel('Fecha')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.show()