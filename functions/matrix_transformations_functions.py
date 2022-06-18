import numpy as np
from scipy.stats import norm


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def compute_empirical_mean(X):
    return mcol(X.mean(1))


def compute_empirical_cov(X):
    mu = compute_empirical_mean(X)
    cov = np.dot((X - mu), (X - mu).T) / float(X.shape[1])
    return cov


def compute_sb(X, L):
    SB = 0
    muG = compute_empirical_mean(X)
    for i in set(list(L)):
        D = X[:, L == i]
        mu = compute_empirical_mean(D)
        SB += D.shape[1] * np.dot((mu - muG), (mu - muG).T)
    return SB / X.shape[1]


def ML_GAU(D):  # used to compute empirical mu and cov
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / float(D.shape[1])
    return mu, C


def gaussianization(D):
    Dg = np.zeros(D.shape)

    for dIdx in range(11):
        i = 0
        for element in D[dIdx, :]:
            Dg[dIdx, i] = ((D[dIdx, :] < element).sum() + 1) / (D[dIdx, :].size + 2)
            Dg[dIdx, i] = norm.ppf(Dg[dIdx, i])
            i += 1
    return Dg
