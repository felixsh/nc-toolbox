import dask.array as da
import numpy as np
from sklearn.decomposition import PCA


def principal_decomp(X, n_components=None, center=False):
    """
    Adapted from sklearn.decompositions.PCA()
    https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/decomposition/_pca.py
    """
    n_samples, n_features = X.shape

    if center:
        column_mean = np.mean(X, axis=0)
        X = X - column_mean
    
    X_dask = da.from_array(X, chunks=(10_000, n_features))
    _, _, Vt = da.linalg.svd(X_dask, coerce_signs=True)
    components = Vt.compute()

    if n_components is None:
        n_components = min(n_samples, n_features) - 1
    elif not 0 <= n_components <= min(n_samples, n_features):
        raise ValueError(f'n_components={n_components} is out of bounds, [0, {min(n_samples, n_features)}]')

    P = components[:n_components, :]
    P_residual = components[n_components:, :]
    
    if center:
        return P, P_residual, column_mean
    else:
        return P, P_residual


if __name__ == '__main__':
    d = 2
    X = np.random.rand(50000, 512)
    P, P_residual, X_mean = principal_decomp(X, n_components=d, center=True)
    
    print(f"==>> X.shape:          {X.shape}")
    print(f"==>> P.shape:          {P.shape}")
    print(f"==>> P_residual.shape: {P_residual.shape}")
    print(np.allclose(P.dot(P_residual.T), 0))

    pca = PCA(n_components=d)
    X_pca = pca.fit_transform(X)
    X_prj = P.dot((X - X_mean).T).T

    print(f"==>> X_pca.shape: {X_pca.shape}")
    print(f"==>> X_prj.shape: {X_prj.shape}")
    print(np.allclose(X_pca, X_prj))
