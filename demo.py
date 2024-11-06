import nc_toolbox as nctb
import numpy as np

# Setup
N = 1000
C = 10
D = 100

W = np.random.rand(C, D)
B = np.random.rand(C)
H = np.random.rand(N, D)
L = np.random.randint(0, C, N)
H_test = np.random.rand(N // 2, D)

# Statistics
mu_c = nctb.class_embedding_means(H, L)
var_c = nctb.class_embedding_variances(H, L, mu_c)
mu_g = nctb.global_embedding_mean(H)

# Metrics
print(f'nc1_strong: {nctb.nc1_strong(H, L, mu_c, mu_g)}')
print(f'nc1_weak: {nctb.nc1_weak(H, L, mu_c, mu_g)}')
print(f'nc1_cdnv: {nctb.nc1_cdnv(mu_c, var_c)}')

print(f'nc2_equinormness: {nctb.nc2_equinormness(mu_c, mu_g)}')
print(f'nc2_equiangularity: {nctb.nc2_equiangularity(mu_c, mu_g)}')
print(f'gnc2_hypershperical_uniformity: {nctb.gnc2_hypershperical_uniformity(mu_c, mu_g)}')

print(f'nc3_self_duality: {nctb.nc3_self_duality(W, mu_c, mu_g)}')
print(f'unc3_uniform_duality: {nctb.unc3_uniform_duality(W, mu_c, mu_g)}')

print(f'nc4_classifier_agreement: {nctb.nc4_classifier_agreement(H_test, W, B, mu_c)}')