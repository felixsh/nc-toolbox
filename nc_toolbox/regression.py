"""This module implements the Neural Regression Collapse metrics from the paper:
Andriopoulos et al. The Prevalence of Neural Collapse in Neural Multivariate Regression, 2024
Notation is taken from the paper.
"""

from collections.abc import Iterable

import numpy as np
from scipy.optimize import minimize

from .decomp import principal_decomp


def _norm(X):
    """Norm rows of matrix X."""
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def _project(C, H):
    """Project rows of H onto subspace defined by columns in C."""
    P = C @ np.linalg.pinv(C)
    return (P @ H.T).T


def nrc1_collapse(H, dim_out):
    """Indicate feature-vector collapse.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned
    by their n principal components.
    Collapse minimizes this metric, goes to zero."""
    M = H.shape[0]
    # TODO This is not centered. Should it be?
    P, _ = principal_decomp(H, n_components=dim_out)
    H_norm = _norm(H)
    x = np.linalg.norm(H_norm - _project(P.T, H_norm), axis=1)
    return float(np.square(x).sum() / M)


def nrc2_duality(H, W):
    """Indicate self-duality.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned by the rows of W.
    Collapse minimizes this metric, goes to zero."""
    M = H.shape[0]
    H_norm = _norm(H)
    x = np.linalg.norm(H_norm - _project(W.T, H_norm), axis=1)
    return float(np.square(x).sum() / M)


def nrc3_structure(W, Sigma, dim_out, gamma=None):
    """Indicate a specific structure of the feature vectors.
    Angles between rows of W are influenced by Sigma. If targets are uncorrelated (Sigma diagonal)
    then NRC3 --> 0 implies W is orthogonal.
    Collapse minimizes this metric, goes to zero."""

    WW_norm = W @ W.T
    WW_norm /= np.linalg.norm(WW_norm, ord='fro')

    L_sigma = np.linalg.cholesky(Sigma)
    In = np.eye(dim_out)

    def f(x):
        gamma_sqrt = np.sqrt(x)[0]
        X_norm = L_sigma - gamma_sqrt * In
        X_norm /= np.linalg.norm(X_norm, ord='fro')
        y = np.linalg.norm(WW_norm - X_norm, ord='fro')
        return np.square(y)

    if gamma is None:
        x0 = np.array([0.5])
        res = minimize(f, x0, method='L-BFGS-B', bounds=((0, 1),))
        return float(res.fun)

    if isinstance(gamma, float):
        x0 = np.array([gamma])
        return float(f(x0))

    if isinstance(gamma, Iterable):
        return [float(f(np.array([g]))) for g in gamma]

    raise ValueError(f'type(gamma): {type(gamma)} is not supported')


def sigma(Y):
    Y_center = Y - Y.mean(axis=0)
    return Y_center.T @ Y_center


if __name__ == '__main__':
    # Test
    n = 1000
    d = 128
    c = 10

    H = np.random.rand(n, d)
    W = np.random.randn(c, d)
    Y = np.random.randn(n, c)

    Sigma = sigma(Y)

    print('nrc1_collapse\t', nrc1_collapse(H, c))
    print('nrc2_duality\t', nrc2_duality(H, W))
    print('nrc3_structure\t', nrc3_structure(W, Sigma, c))

    # print('type(nrc1_collapse)\t', type(nrc1_collapse(H, c)))
    # print('type(nrc2_duality)\t', type(nrc2_duality(H, W)))
    # print('type(nrc3_structure)\t', type(nrc3_structure(W, Sigma, c, gamma=0.5)))

    # Plot NRC3 over gamma
    import matplotlib.pyplot as plt

    gamma = np.linspace(0, 1, 101)
    nrc3 = nrc3_structure(W, Sigma, c, gamma)

    plt.plot(gamma, nrc3)
    plt.show()
