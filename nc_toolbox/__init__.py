from .statistic import class_embedding_means, class_embedding_variances
from .statistic import global_embedding_mean
from .statistic import split_embeddings, center_embeddings
from .statistic import between_class_covariance, within_class_covariance

from .metrics import nc1_strong, nc1_weak, nc1_cdnv
from .metrics import nc2_equinormness, nc2_equiangularity, gnc2_hypershperical_uniformity
from .metrics import nc3_self_duality, unc3_uniform_duality
from .metrics import nc4_classifier_agreement

from .decomp import principal_decomp

from .util import lin_classify, ncc_classify