"""This module implements the Neural Regression Collapse metrics from the paper:
Andriopoulos et al. The Prevalence of Neural Collapse in Neural Multivariate Regression, 2024
Notation is taken from the paper.
"""

from collections.abc import Iterable
from typing import Union

import dask.array as da
import numpy as np
from .decomp import principal_decomp
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.utils.extmath import svd_flip


def _rowwise_norm(X: NDArray) -> NDArray:
    """Norm rows of matrix X."""
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def _proj(H: NDArray, C: NDArray) -> NDArray:
    """Project rows of H onto subspace defined by columns in C."""
    P = C @ np.linalg.pinv(C)
    return (P @ H.T).T


def _proj_orth(H: NDArray, C: NDArray) -> NDArray:
    """Project rows of H onto subspace defined by columns in C, C is orthogonal."""
    P = C @ C.T
    return (P @ H.T).T


def nrc1_collapse(H: NDArray, dim_out: int) -> float:
    """Indicate feature-vector collapse.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned
    by their n principal components.
    Collapse minimizes this metric, goes to zero."""
    P, _ = principal_decomp(H, n_components=dim_out)
    H_norm = _rowwise_norm(H)
    x = np.linalg.norm(H_norm - _proj_orth(H_norm, P.T), axis=1, ord=2)
    return float(np.square(x).mean())


def nrc1_collapse_all(H: NDArray) -> NDArray:
    """Indicate feature-vector collapse.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned
    by their n principal components.
    Collapse minimizes this metric, goes to zero."""
    _, dim_feature = H.shape
    P, _ = principal_decomp(H, n_components=dim_feature)
    H_norm = _rowwise_norm(H)
    res = np.empty((dim_feature,))
    Id = np.eye(dim_feature)
    for n in range(1, dim_feature + 1):
        C = P[:n, :].T @ P[:n, :]
        res[n - 1] = np.square(np.linalg.norm(H_norm @ (Id - C), axis=1, ord=2)).mean()
    return res


def nrc2_duality(H: NDArray, W: NDArray) -> float:
    """Indicate self-duality.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned by the rows of W.
    Collapse minimizes this metric, goes to zero."""
    M = H.shape[0]
    H_norm = _rowwise_norm(H)
    x = np.linalg.norm(H_norm - _proj(H_norm, W.T), axis=1)
    return float(np.square(x).sum() / M)


def nrc3_structure(
    W: NDArray, Y: NDArray, dim_out: int, gamma: Union[float, Iterable, None] = None
) -> float:
    """Indicate a specific structure of the feature vectors.
    Angles between rows of W are influenced by Sigma. If targets are uncorrelated (Sigma diagonal)
    then NRC3 --> 0 implies W is orthogonal.
    Collapse minimizes this metric, goes to zero."""
    M, _ = Y.shape

    WW_norm = W @ W.T
    WW_norm /= np.linalg.norm(WW_norm, ord='fro')

    Y_centered = Y - Y.mean(axis=1)[:, np.newaxis]
    Y_dask = da.from_array(Y_centered.T, chunks='auto')
    U, Sigma_, Vt = da.linalg.svd(Y_dask, coerce_signs=True)
    U = U.compute()
    Sigma_.compute()
    Vt = Vt.compute()
    U, _ = svd_flip(U, Vt)
    Sigma_sqrt = U @ Sigma_ / np.sqrt(M) @ U.T

    In = np.eye(dim_out)

    def f(x: NDArray) -> np.float64:
        gamma_sqrt = np.sqrt(x)[0]
        X_norm = Sigma_sqrt - gamma_sqrt * In
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


if __name__ == '__main__':
    # Test
    n_sample = 1000
    d_feature = 128
    n_classes = 10

    H = np.random.rand(n_sample, d_feature)
    W = np.random.randn(n_classes, d_feature)
    Y = np.random.randn(n_sample, n_classes)

    print('nrc1_collapse\t', nrc1_collapse(H, n_classes))
    print('nrc2_duality\t', nrc2_duality(H, W))
    print('nrc3_structure\t', nrc3_structure(W, Y, n_classes))

    print('nrc1_collapse_all\t', nrc1_collapse_all(H))

    # print('type(nrc1_collapse)\t', type(nrc1_collapse(H, c)))
    # print('type(nrc2_duality)\t', type(nrc2_duality(H, W)))
    # print('type(nrc3_structure)\t', type(nrc3_structure(W, Sigma, c, gamma=0.5)))

    # Plot NRC3 over gamma
    import matplotlib.pyplot as plt

    # gamma = np.linspace(0, 1, 101)
    # nrc3 = nrc3_structure(W, Y, n_classes, gamma)
    # plt.plot(gamma, nrc3)
    # plt.show()

    plt.plot(nrc1_collapse_all(H))
    plt.show()
