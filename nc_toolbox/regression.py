"""This module implements the Neural Regression Collapse metrics from the paper:
Andriopoulos et al. The Prevalence of Neural Collapse in Neural Multivariate Regression, 2024
"""

import numpy as np
from decomp import principal_decomp, project


def _norm(X):
    return X / np.linalg.norm(X, axis=1)


def nrc1_feature_collapse(H, dim_out):
    """Indicates feature-vector collapse.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned by their n principal components."""
    M = H.shape[0]
    # TODO This is not centered. Should it be?
    P, _ = principal_decomp(H, n_components=dim_out)
    H_norm = _norm(H)
    x = np.linalg.norm(H_norm - project(P, H_norm), axis=1)
    return np.square(x).sum() / M


def nrc2_self_duality(H, W, dim_out):
    """Indicates self-duality.
    The d-dim feature vectors collapse to a much lower n-dim subspace spanned by the rows of W."""
    M = H.shape[0]
    # TODO This is not centered. Should it be?
    P, _ = principal_decomp(W, n_components=dim_out)
    H_norm = _norm(H)
    x = np.linalg.norm(H_norm - project(P, H_norm), axis=1)
    return np.square(x).sum() / M


def nrc3_structure():
    pass
