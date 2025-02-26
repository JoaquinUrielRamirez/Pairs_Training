import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generar_senales_trading(mu, std_threshold=1.5):
    """
    Genera señales de trading basadas en el término de corrección de error.

    Parámetros:
    - mu: DataFrame con el término de corrección de error.
    - std_threshold: Umbral en desviaciones estándar para generar señales.

    Retorna:
    - DataFrame con señales de trading.
    """
    mu_mean = mu.mean()
    mu_std = mu.std()
    upper_bound = mu_mean + std_threshold * mu_std
    lower_bound = mu_mean - std_threshold * mu_std

    señales = pd.DataFrame(index=mu.index)

    # Seleccionar la primera columna de mu si tiene múltiples columnas
    mu_selected = mu.iloc[:, 0] if isinstance(mu, pd.DataFrame) else mu

    señales['Long'] = (mu_selected < lower_bound.iloc[0]).astype(int)
    señales['Short'] = (mu_selected > upper_bound.iloc[0]).astype(int)

    return señales


def backtest_estrategia(mu, std_threshold=1.5):
    """
    Realiza un backtest de la estrategia basada en el término de corrección de error.

    Parámetros:
    - mu: DataFrame con el término de corrección de error.
    - std_threshold: Umbral en desviaciones estándar para abrir posiciones.

    Retorna:
    - DataFrame con el rendimiento acumulado de la estrategia.
    """
    señales = generar_senales_trading(mu, std_threshold)

    # Simulación de rendimiento: ir long cuando 'Long' == 1, ir short cuando 'Short' == 1
    retornos = señales['Long'] - señales['Short']
    retorno_acumulado = retornos.cumsum()

    return retorno_acumulado


def visualizar_senales(mu, señales):
    """
    Genera un gráfico que muestra el término de corrección de error y las señales de trading.

    Parámetros:
    - mu: DataFrame con el término de corrección de error.
    - señales: DataFrame con señales de trading.
    """
    plt.figure(figsize=(12,6))
    plt.plot(mu.iloc[:, 0], label='Término de Corrección de Error', color='blue')

    # Dibujar líneas horizontales para los umbrales
    mu_mean = mu.mean()
    mu_std = mu.std()
    upper_bound = mu_mean + 1.5 * mu_std
    lower_bound = mu_mean - 1.5 * mu_std

    plt.axhline(upper_bound.iloc[0], color='red', linestyle='dashed', label='+1.5 std')
    plt.axhline(lower_bound.iloc[0], color='green', linestyle='dashed', label='-1.5 std')
    plt.axhline(mu_mean.iloc[0], color='black', linestyle='dashed', label='Media')

    # Corregimos las señales de trading
    plt.scatter(señales.index[señales['Long'] == 1], mu.iloc[:, 0][señales['Long'] == 1], color='green', marker='^', label='Long', alpha=1)
    plt.scatter(señales.index[señales['Short'] == 1], mu.iloc[:, 0][señales['Short'] == 1], color='red', marker='v', label='Short', alpha=1)

    plt.legend()
    plt.xlabel('Fecha')
    plt.ylabel('Término de Corrección de Error')
    plt.title('Señales de Trading basadas en VECM')
    plt.grid()
    plt.show()


def ajustar_estrategia_balanceada(mu, hedge_ratio, std_threshold=1.5):
    """
    Ajusta la estrategia de trading para balancear mejor las posiciones long y short.

    Parámetros:
    - mu: DataFrame con el término de corrección de error.
    - hedge_ratio: Serie con el hedge ratio dinámico.
    - std_threshold: Umbral en desviaciones estándar para generar señales.

    Retorna:
    - DataFrame con señales de trading balanceadas.
    """
    mu_mean = mu.mean()
    mu_std = mu.std()
    upper_bound = mu_mean + std_threshold * mu_std
    lower_bound = mu_mean - std_threshold * mu_std

    señales = pd.DataFrame(index=mu.index)

    # Seleccionar la primera columna de mu si tiene múltiples columnas
    mu_selected = mu.iloc[:, 0] if isinstance(mu, pd.DataFrame) else mu

    señales['Long'] = (mu_selected < lower_bound.iloc[0]).astype(int)
    señales['Short'] = (mu_selected > upper_bound.iloc[0]).astype(int)

    # Ajustar posiciones balanceadas usando el hedge ratio
    señales['Long_Size'] = señales['Long'] * hedge_ratio
    señales['Short_Size'] = señales['Short'] * hedge_ratio

    return señales