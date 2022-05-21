from matrix_transformations import *


def ML_GAU(D):              # used to compute empirical mu and cov
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / float(D.shape[1])
    return mu, C