import numpy as np
from math import isclose
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr

# https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables


def create_correlated_array(arr, corr,
                            corr_method=pearsonr,
                            abs_tol=1e-5):
    n = len(arr)

    theta = np.arccos(corr)

    x_1 = arr.copy()
    x_2 = np.random.normal(loc=0, scale=1, size=n)

    X = np.column_stack([x_1, x_2])
    X_stand = StandardScaler(with_std=False).fit_transform(X)
    identity_mat = np.identity(n)
    Q, _ = qr(X_stand[:, [0]], mode='economic')
    P = np.dot(Q, Q.T)
    x2_ortho = np.dot(identity_mat - P, X_stand[:, [1]])
    Xc2 = np.column_stack([X_stand[:, [0]], x2_ortho])
    Y_diagonals = np.zeros([2, 2])
    diagonals = 1/np.sqrt(np.sum(Xc2**2, axis=0))
    np.fill_diagonal(Y_diagonals, diagonals)  # inplace
    Y = np.dot(Xc2, Y_diagonals)

    x = Y[:, [1]] + (1 / np.tan(theta)) * Y[:, [0]]

    x = x.reshape(-1)

    # checking whether result is according to expectations
    isclose(corr_method(x, x_1)[0], corr, abs_tol=abs_tol)

    return x
