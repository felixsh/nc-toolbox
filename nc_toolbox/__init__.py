"""
N: number of samples
Nc: number of samples in class c
D: feature dimensionality
C: number of classes

W: weight matrix of linear classifier (C, D)
H: feature matrix (N, D)
L: label vector (N,)

mu_c: matrix of class embedding means (C, D)
mu_g: vector of global embedding mean (D,)
"""

from .statistic import class_embedding_means, class_embedding_variances
from .statistic import global_embedding_mean

from .metrics import nc1_cdnv
from .metrics import nc2_equinormness, nc2_equiangularity, gnc2_hypershperical_uniformity
from .metrics import nc3_self_duality, unc3_uniform_duality
from .metrics import nc4_classifier_agreement
