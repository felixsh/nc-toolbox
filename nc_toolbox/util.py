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


def lin_classify(H, W, b):
    """
    Linear classifier
    """
    return np.argmax((H @ W.T) + b, axis=1)


def _ncc_classify(h, mu_c):
    """
    h: feature vector (D,)
    """
    dist = np.linalg.norm(mu_c - h, axis=1)
    return np.argmin(dist)


def ncc_classify(H, mu_c):
    """
    Nearest Class Center (NCC) classifier
    """
    return np.apply_along_axis(_ncc_classify, 1, H, mu_c)


def testrun_classifiers():
    n = 1000
    d = 128
    c = 10
    H = np.random.rand(n, d)
    W = np.random.rand(c, d)
    b = np.random.rand(c)
    mu_c = np.random.rand(c, d)
    lin_res = lin_classify(H, W, b)
    print(f"==>> lin_res.shape: {lin_res.shape}")
    ncc_res = ncc_classify(H, mu_c)
    print(f"==>> ncc_res.shape: {ncc_res.shape}")


if __name__ == '__main__':
    testrun_classifiers()
