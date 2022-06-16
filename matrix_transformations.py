import numpy as np


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def compute_empirical_cov(X):
    mu = compute_empirical_mean(X)
    cov = np.dot((X - mu), (X - mu).T) / X.shape
    return cov


def compute_empirical_mean(X):
    return mcol(X.mean(1))


def compute_sb(X, L):
    SB = 0
    muG = compute_empirical_mean(X)
    for i in set(list(L)):
        D = X[:, L == i]
        mu = compute_empirical_mean(D)
        SB += D.shape[1] * np.dot((mu - muG), (mu - muG).T)
    return SB / X.shape[1]
