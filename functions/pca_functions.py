from .matrix_transformations_functions import *


def pca(D, L, m):
    # mu = compute_empirical_mean(D)  # gives the same result as loop and division but faster
    C = compute_empirical_cov(D)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP
