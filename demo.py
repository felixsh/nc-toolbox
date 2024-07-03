import numpy as np

from statistic import class_embedding_means, class_embedding_variances, global_embedding_mean
from metrics import nc1_cdnv
from metrics import nc2_equinormness, nc2_equiangularity, gnc2_hypershperical_uniformity
from metrics import nc3_self_duality, unc3_uniform_duality
from metrics import nc4_classifier_agreement


if __name__ == '__main__':
    N = 1000
    C = 10
    D = 100

    W = np.random.rand(C, D)
    H = np.random.rand(N, D)
    L = np.random.randint(0, C, N)
    H_test = np.random.rand(N // 2, D)

    mu_c = class_embedding_means(H, L)
    var_c = class_embedding_variances(H, L, mu_c)
    mu_g = global_embedding_mean(H)

    print(f'nc1_cdnv: {nc1_cdnv(mu_c, mu_g)}')

    print(f'nc2_equinormness: {nc2_equinormness(mu_c, mu_g)}')
    print(f'nc2_equiangularity: {nc2_equiangularity(mu_c, mu_g)}')
    print(f'gnc2_hypershperical_uniformity: {gnc2_hypershperical_uniformity(mu_c, mu_g)}')

    print(f'nc3_self_duality: {nc3_self_duality(W, mu_c, mu_g)}')
    print(f'unc3_uniform_duality: {unc3_uniform_duality(W, mu_c, mu_g)}')

    print(f'nc4_classifier_agreement: {nc4_classifier_agreement(H_test, W, mu_c)}')
