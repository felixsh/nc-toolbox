from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .statistic import between_class_covariance, within_class_covariance
from .util import lin_classify, ncc_classify, reduce_func, triu


def nc1_strong(H: NDArray, L: NDArray, mu_c: NDArray, mu_g: NDArray) -> float:
    """
    Calculate class-features variance collapse
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training, 2020, Fig. 6
    Collapse minimizes this metric, goes to zero.
    """
    C = mu_c.shape[0]
    sigma_b = between_class_covariance(mu_c, mu_g)
    sigma_w = within_class_covariance(H, L, mu_c)
    return float(np.trace((sigma_w @ np.linalg.pinv(sigma_b)) / C))


def nc1_weak(H: NDArray, L: NDArray, mu_c: NDArray, mu_g: NDArray) -> tuple[float]:
    """
    Calculate normalized class variances
    Rangamani, Akshay, et al. Feature learning in deep classifiers through intermediate neural collapse, 2023, §4
    Collapse minimizes within class variance and maximizes between class variance.
    """
    tr_sigma_b = np.trace(between_class_covariance(mu_c, mu_g))
    tr_sigma_w = np.trace(within_class_covariance(H, L, mu_c))
    tr_sigma_tot = tr_sigma_b + tr_sigma_w
    nomalized_between_class_var = tr_sigma_b / tr_sigma_tot
    nomalized_within_class_var = tr_sigma_w / tr_sigma_tot
    return float(nomalized_between_class_var), float(nomalized_within_class_var)


def nc1_cdnv(mu_c: NDArray, var_c: NDArray, reduction: Optional[str] = 'mean') -> float:
    """
    Class-Distance Normalized Variance (CDNV) / Within-Class Variability Collapse
    Tomer Galanti, András György, and Marcus Hutter. On the role of neural collapse in transfer learning, 2022
    Collapse minimizes this metric, goes to zero.
    """
    C = mu_c.shape[0]
    idx0, idx1 = np.triu_indices(C, k=1)
    num = var_c[idx0] + var_c[idx1]
    den = 2 * np.square(mu_c[idx0] - mu_c[idx1]).sum(axis=1)
    cdnv = num / den
    return float(reduce_func(cdnv, reduction))


def nc2_equinormness(
    mu_c: NDArray, mu_g: NDArray, reduction: Optional[str] = 'mean'
) -> float:
    """
    Coefficient of variation for log distances of mean class embeddings to global mean of embeddings.
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.5
    Collapse minimizes this metric, goes to zero.
    """
    C = mu_c.shape[0]
    idx0, idx1 = np.triu_indices(C, k=1)
    mu_centered = mu_c - mu_g
    log_dist = np.log(np.linalg.norm(mu_centered, axis=1))  # (C,)
    diff = log_dist[idx0] - log_dist[idx1]
    return float(reduce_func(diff, reduction))


def nc2_equiangularity(
    mu_c: NDArray, mu_g: NDArray, reduction: Optional[str] = 'mean'
) -> float:
    """
    Coefficient of variation for cosine similarity between class means minus expected value.
    Also called interference.
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.5
    Collapse minimizes this metric, goes to zero.
    """
    C = mu_c.shape[0]
    expected_angle = -1 / (C - 1)
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    cossim = triu(mu_centered_norm @ mu_centered_norm.T, k=1)
    return float(reduce_func(cossim - expected_angle, reduction))


def gnc2_hypershperical_uniformity(
    mu_c: NDArray, mu_g: NDArray, reduction: Optional[str] = 'mean'
) -> float:
    """
    Relaxation from the ETF structure, measures generalized neural collapse as pair-wise interactions under a logarithmic inverse distance kernel.
    Jianfeng Lu and Stefan Steinerberger. Neural collapse with cross-entropy loss, 2021
    Weiyang Liu, Longhui Yu, Adrian Weller, and Bernhard Schölkopf. Generalizing and decoupling neural collapse via hyperspherical uniformity gap, 2023
    TODO Collapse minimizes this metric, goes to zero??
    """
    C = mu_c.shape[0]
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    idx0, idx1 = np.triu_indices(C, k=1)
    diff = mu_centered_norm[idx0] - mu_centered_norm[idx1]
    dist = np.linalg.norm(diff, axis=1)  # TODO norm not specified in source!
    log_inv_dist = np.log(np.reciprocal(dist))
    return float(reduce_func(log_inv_dist, reduction))


def nc3_self_duality(W: NDArray, mu_c: NDArray, mu_g: NDArray) -> float:
    """
    Measures alignment between class means and linear classifier weights.
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, sep 2020
    Collapse minimizes this metric, goes to zero.
    """
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm = mu_centered / np.linalg.norm(mu_centered)
    W_norm = W / np.linalg.norm(W)  # (C, D)
    return float(np.linalg.norm(W_norm - mu_centered_norm))


def unc3_uniform_duality(
    W: NDArray, mu_c: NDArray, mu_g: NDArray, reduction: Optional[str] = 'mean'
) -> float:
    """
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.6
    """
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    W_norm = W / np.linalg.norm(W, axis=1)[:, None]  # (C, D)
    cossim = triu(W_norm @ mu_centered_norm.T, k=1)
    return float(reduce_func(cossim, reduction))


def nc4_classifier_agreement(
    H: NDArray, W: NDArray, b: NDArray, mu_c: NDArray
) -> float:
    """
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.7
    """
    lin_res = lin_classify(H, W, b)
    ncc_res = ncc_classify(H, mu_c)
    agreement = lin_res == ncc_res
    return float(np.mean(agreement))
