import numpy as np


def triu(A, k=0):
    """
    Return upper triangle of square matrix as flat vector
    A: square matrix
    k=0: with main diagonal
    k=1: upper triangle without main diagonal
    """
    return A[np.triu_indices_from(A, k=k)].flatten()


def cov(x):
    """
    Coefficient Of Variation (COV)
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024
    x: vector of values
    """
    return x.std() / x.mean()


def ncc_classify(h, mu_c):
    """
    Nearest Class Center (NCC) classifier
    h: feature vector (D,)
    """
    dist = np.linalg.norm(mu_c - h, axis=1)
    return np.argmin(dist)
