"""A package for calculating Neural Collapse and related metrics."""

from .decomp import principal_decomp
from .metrics import (
    gnc2_hypershperical_uniformity,
    nc1_cdnv,
    nc1_strong,
    nc1_weak,
    nc2_equiangularity,
    nc2_equinormness,
    nc3_self_duality,
    nc4_classifier_agreement,
    unc3_uniform_duality,
)
from .regression import nrc1_feature_collapse, nrc2_self_duality
from .statistic import (
    between_class_covariance,
    center_embeddings,
    class_embedding_means,
    class_embedding_variances,
    global_embedding_mean,
    split_embeddings,
    within_class_covariance,
)
from .util import lin_classify, ncc_classify

__all__ = [
    'principal_decomp',
    'gnc2_hypershperical_uniformity',
    'nc1_cdnv',
    'nc1_strong',
    'nc1_weak',
    'nc2_equiangularity',
    'nc2_equinormness',
    'nc3_self_duality',
    'nc4_classifier_agreement',
    'unc3_uniform_duality',
    'between_class_covariance',
    'center_embeddings',
    'class_embedding_means',
    'class_embedding_variances',
    'global_embedding_mean',
    'split_embeddings',
    'within_class_covariance',
    'lin_classify',
    'ncc_classify',
    'nrc1_feature_collapse',
    'nrc2_self_duality',
]
