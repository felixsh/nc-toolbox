from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray


def class_embedding_means(H: NDArray, L: NDArray) -> NDArray:
    """Calculate means of embeddings for every class."""
    unique_c = np.unique(L)
    C = len(unique_c)
    D = H.shape[1]
    mu_c = np.empty((C, D))
    for i, c in enumerate(unique_c):
        idx = L == c
        mu_c[i, :] = H[idx, :].mean(axis=0)
    return mu_c


def class_embedding_variances(H: NDArray, L: NDArray, mu_c: NDArray) -> NDArray:
    """
    Calculate variances of embeddings squared norm for every class.
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.2
    """
    unique_c = np.unique(L)
    C = len(unique_c)
    var_c = np.empty((C,))
    for i, c in enumerate(unique_c):
        idx = L == c
        square_dist = np.square(H[idx, :] - mu_c[c, :]).sum(axis=1)
        Nc = square_dist.shape[0]
        var_c[i] = square_dist.sum(axis=0) / (Nc - 1)
    return var_c


def global_embedding_mean(H: NDArray) -> NDArray:
    """Calculate mean of embeddings globally."""
    mu_g = H.mean(axis=0)
    return mu_g


def global_embedding_variance(H: NDArray, mu_g: NDArray) -> float:
    """Calculate variance of embeddings globally."""
    square_dist = np.square(H - mu_g).sum(axis=1)
    N = square_dist.shape[0]
    var_g = square_dist.sum(axis=0) / (N - 1)
    return float(var_g)


def split_embeddings(H: NDArray, L: NDArray) -> dict:
    """Split class embeddings into arrays per class."""
    class_labels = np.unique(L)
    H_splitted = {}
    for c in class_labels:
        idx = L == c
        H_splitted[c] = H[idx, :]
    return H_splitted


def center_embeddings(H: NDArray, L: NDArray, mu_c: NDArray) -> NDArray:
    """Center embeddings on their respective class mean."""
    class_labels = np.unique(L)
    H_centered = np.empty_like(H)
    for c in class_labels:
        idx = L == c
        H_centered[idx, :] = H[idx, :] - mu_c[c, :]
    return H_centered


def between_class_covariance(mu_c: NDArray, mu_g: NDArray) -> NDArray:
    """
    Calculate between class covariance.
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training, 2020, §I
    """
    C = mu_c.shape[0]
    diff = mu_c - mu_g
    sigma_b = (diff.T @ diff) / C
    return sigma_b


def within_class_covariance(H: NDArray, L: NDArray, mu_c: NDArray) -> NDArray:
    """
    Calculate within class covariance.
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training, 2020, §I
    """
    N = H.shape[0]
    H_center = center_embeddings(H, L, mu_c)
    sigma_w = (H_center.T @ H_center) / N
    return sigma_w
