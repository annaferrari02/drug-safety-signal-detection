import numpy as np
from scipy.special import gdtr


def f_cost_quantiles(p, threshold, Q, a1, b1, a2, b2):
    one = Q * gdtr(p, a1, b1)
    two = (1 - Q) * gdtr(p, a2, b2)
    one = np.where(np.isnan(one), 0.0, one)
    two = np.where(np.isnan(two), 0.0, two)
    return one + two - threshold


def quantiles(threshold, Q, a1, b1, a2, b2):
    """
    Calculate CI lower bound using algorithms from DuMouchel (1999).
    Versione vettorizzata corretta per array numpy.
    """
    Q  = np.asarray(Q,  dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    length = Q.shape[0] if Q.ndim > 0 else 1

    m = np.full(length, -100000.0)
    M = np.full(length,  100000.0)
    x = np.ones(length)

    for _ in range(200):  # max iterazioni invece di while infinito
        cost = f_cost_quantiles(x, threshold, Q, a1, b1, a2, b2)
        if np.max(np.abs(np.round(cost * 1e4))) == 0:
            break
        S = np.sign(cost)
        xnew = (1 + S) / 2 * ((x + m) / 2) + (1 - S) / 2 * ((M + x) / 2)
        M = (1 + S) / 2 * x + (1 - S) / 2 * M
        m = (1 + S) / 2 * m + (1 - S) / 2 * x
        x = xnew

    return x
