import numpy


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))

def vrow(v):
    return v.reshape((1, v.size))


def compute_empirical_cov(X):
    mu = compute_empirical_mean(X)
    cov = numpy.dot((X - mu), (X - mu).T) / X.shape[1]
    return cov


def compute_empirical_mean(X):
    return mcol(X.mean(1))


def compute_sb(X, L):
    SB = 0
    muG = compute_empirical_mean(X)
    for i in set(list(L)):
        D = X[:, L == i]
        mu = compute_empirical_mean(D)
        # print(i)
        # print((mu - muG) * D.shape[1])
        # print(mu)
        # print(muG)
        # print()
        SB += D.shape[1] * numpy.dot((mu - muG), (mu - muG).T)
    return SB / X.shape[1]


