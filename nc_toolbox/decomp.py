from typing import Optional

import dask.array as da
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def svd_flip(v: NDArray) -> NDArray:
    """Sign correction to ensure deterministic output from SVD.

    Adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/utils/extmath.py#L848
    """
    max_abs_v_rows = np.argmax(np.abs(v), axis=1)
    shift = np.arange(v.shape[0])
    indices = max_abs_v_rows + shift * v.shape[1]
    signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
    v *= signs[:, np.newaxis]
    return v


def project(P: NDArray, X: NDArray, X_mean: Optional[NDArray] = None) -> NDArray:
    """Project features X onto subspace defined by projection matrix P."""
    if X_mean is None:
        return P.dot(X.T).T
    else:
        return P.dot((X - X_mean).T).T


def principal_decomp(
    X: NDArray, n_components: Optional[int] = None, center: Optional[bool] = False
) -> tuple[NDArray]:
    """Principle component decomposition with Dask.

    Adapted from sklearn.decompositions.PCA():
    https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/decomposition/_pca.py
    """
    n_samples, n_features = X.shape

    if center:
        column_mean = np.mean(X, axis=0)
        X = X - column_mean

    X_dask = da.from_array(X, chunks=(10_000, n_features))
    _, _, Vt = da.linalg.svd(X_dask, coerce_signs=True)
    components = Vt.compute()
    components = svd_flip(components)

    if n_components is None:
        n_components = min(n_samples, n_features) - 1
    elif not 0 <= n_components <= min(n_samples, n_features):
        raise ValueError(
            f'n_components={n_components} is out of bounds, [0, {min(n_samples, n_features)}]'
        )

    P = components[:n_components, :]
    P_residual = components[n_components:, :]

    if center:
        return P, P_residual, column_mean
    else:
        return P, P_residual


if __name__ == '__main__':
    # Basic testing

    d = 10
    X = np.random.rand(10000, 512)
    P_dask, P_residual, X_mean = principal_decomp(X, n_components=d, center=True)
    X_dask = project(P_dask, X, X_mean)

    print(f'==>> X.shape:          {X.shape}')
    print(f'==>> P_dask.shape:     {P_dask.shape}')
    print(f'==>> P_residual.shape: {P_residual.shape}')
    print()

    pca = PCA(n_components=d)
    X_pca = pca.fit_transform(X)
    P_pca = pca.components_

    _, _, Vh = np.linalg.svd(X - X_mean, full_matrices=True)
    Vh = svd_flip(Vh)
    P_np = Vh[:d, :]
    X_np = P_np.dot((X - X_mean).T).T

    print(f'==>> P_dask.shape:     {P_dask.shape}')
    print(f'==>> P_pca.shape:      {P_pca.shape}')
    print(f'==>> P_np.shape:       {P_np.shape}')
    print(np.allclose(P_pca, P_dask))
    print(np.allclose(P_pca, P_np))
    print(np.allclose(P_np, P_dask))
    print(np.allclose(P_dask.dot(P_residual.T), 0))
    print(np.allclose(P_pca.dot(P_residual.T), 0))
    print(np.allclose(P_np.dot(P_residual.T), 0))
    print()

    print(f'==>> X_pca.shape:  {X_pca.shape}')
    print(f'==>> X_dask.shape: {X_dask.shape}')
    print(f'==>> X_np.shape:   {X_np.shape}')
    print(np.allclose(X_pca, X_dask))
    print(np.allclose(X_pca, X_np))
    print(np.allclose(X_np, X_dask))
