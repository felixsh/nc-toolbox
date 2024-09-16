import numpy as np

from nc_toolbox.util import triu, cov, lin_classify, ncc_classify


def nc1_strong():
    """
    TODO
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, sep 2020
    """
    pass


def nc1_weak():
    """
    TODO
    """
    pass


def nc1_cdnv(mu_c, var_c):
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
    return cov(cdnv)


def nc2_equinormness(mu_c, mu_g):
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
    return cov(diff)


def nc2_equiangularity(mu_c, mu_g):
    """
    Coefficient of variation for cosine similarity between class means minus expected value.
    Also called interference.
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.5
    Collapse minimizes this metric, goes to zero.
    """
    C = mu_c.shape[0]
    expected_angle = -1 / (C - 1)
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm  = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    cossim = triu(mu_centered_norm @ mu_centered_norm.T, k=1)
    return cov(cossim - expected_angle)


def gnc2_hypershperical_uniformity(mu_c, mu_g):
    """
    Relaxation from the ETF structure, measures generalized neural collapse as pair-wise interactions under a logarithmic inverse distance kernel.
    Jianfeng Lu and Stefan Steinerberger. Neural collapse with cross-entropy loss, 2021
    Weiyang Liu, Longhui Yu, Adrian Weller, and Bernhard Schölkopf. Generalizing and decoupling neural collapse via hyperspherical uniformity gap, 2023
    TODO Collapse minimizes this metric, goes to zero??
    """
    C = mu_c.shape[0]
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm  = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    idx0, idx1 = np.triu_indices(C, k=1)
    diff = mu_centered_norm[idx0] - mu_centered_norm[idx1]
    dist = np.linalg.norm(diff, axis=1)  # TODO norm not specified in source!
    log_inv_dist = np.log(np.reciprocal(dist))
    return cov(log_inv_dist)


def nc3_self_duality(W, mu_c, mu_g):
    """
    Measures alignment between class means and linear classifier weights.
    Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, sep 2020
    Collapse minimizes this metric, goes to zero.
    """
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm  = mu_centered / np.linalg.norm(mu_centered)
    W_norm  = W / np.linalg.norm(W)  # (C, D)
    return np.linalg.norm(W_norm - mu_centered_norm)


def unc3_uniform_duality(W, mu_c, mu_g):
    """
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.6
    """
    mu_centered = mu_c - mu_g  # (C, D)
    mu_centered_norm  = mu_centered / np.linalg.norm(mu_centered, axis=1)[:, None]
    W_norm = W / np.linalg.norm(W, axis=1)[:, None]  # (C, D)
    cossim = triu(W_norm @ mu_centered_norm.T, k=1)
    return cov(cossim)


def nc4_classifier_agreement(H, W, b, mu_c):
    """
    Robert Wu and Vardan Papyan. Linguistic Collapse: Neural Collapse in (Large) Language Models, 2024, §3.7
    """
    lin_res = lin_classify(H, W, b)
    ncc_res = ncc_classify(H, mu_c)
    agreement = lin_res == ncc_res
    return np.mean(agreement)
