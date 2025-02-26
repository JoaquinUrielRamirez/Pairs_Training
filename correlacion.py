import numpy as np

def corr(corr_matrix):
    np.fill_diagonal(corr_matrix.values, np.nan)  # Remplaza la diagonal con NAN
    max_corr_value = corr_matrix.max().max()
    max_corr_pair = corr_matrix.stack().idxmax()  # Encuentra las variables con mayor correlación

    return max_corr_value, max_corr_pair
