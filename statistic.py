import numpy as np


def class_embedding_means(H, L):
    """Calculate means of embeddings for every class."""
    unique_c = np.unique(L)
    C = len(unique_c)
    D = H.shape[1]
    mu_c = np.empty((C, D))
    for c in unique_c:
        idx = L == c
        mu_c[c, :] = H[idx, :].mean(axis=0)
    return mu_c


def class_embedding_variances(H, L, mu_c):
    """
    Calculate variances of embeddings squared norm for every class.
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.2
    """
    unique_c = np.unique(L)
    C = len(unique_c)
    var_c = np.empty((C,))
    for c in unique_c:
        idx = L == c
        square_dist = np.square(H[idx, :] - mu_c[c, :]).sum(axis=1)
        Nc = square_dist.shape[0]
        var_c[c] = square_dist.sum(axis=0) / (Nc - 1)
    return var_c


def global_embedding_mean(H):
    """Calculate mean of embeddings globally."""
    mu_g = H.mean(axis=0)
    return mu_g
