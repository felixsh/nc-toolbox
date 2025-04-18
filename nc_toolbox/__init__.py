"""A package for calculating Neural Collapse and related metrics."""

from .decomp import principal_decomp, project, project_keepdim
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
from .regression import nrc1_collapse, nrc1_collapse_all, nrc2_duality, nrc3_structure
from .statistic import (
    between_class_covariance,
    center_embeddings,
    class_embedding_means,
    class_embedding_variances,
    global_embedding_mean,
    global_embedding_variance,
    split_embeddings,
    within_class_covariance,
)
from .util import lin_classify, ncc_classify, triu

__all__ = [
    'principal_decomp',
    'project',
    'project_keepdim',
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
    'global_embedding_variance',
    'split_embeddings',
    'within_class_covariance',
    'lin_classify',
    'ncc_classify',
    'triu',
    'nrc1_collapse',
    'nrc1_collapse_all',
    'nrc2_duality',
    'nrc3_structure',
]
