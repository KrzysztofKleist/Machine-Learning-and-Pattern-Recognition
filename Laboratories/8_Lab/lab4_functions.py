import numpy as np
import numpy.linalg.linalg


def logpdf_1sample(x, mu, C):
    P = np.linalg.inv(C)
    res = -0.5 * x.shape[0] * np.log(2*np.pi)
    res += -0.5 * np.linalg.slogdet(C)[1]
    res += -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
    return res.ravel()


def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


def logpdf_GAU_ND_2(X, mu, C):
    P = np.linalg.inv(C)
    res = -0.5*X.shape[0] * np.log(2*np.pi) + 0.5*np.linalg.slogdet(P)[1]
    res = res - 0.5 * ((X-mu)*np.dot(P, (X-mu))).sum[0]
    return res


def logpdf_GAU_ND_Opt(X, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * X.shape[0] * np.log(2*np.pi)
    const += -0.5 * np.linalg.slogdet(C)[1]
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    return np.array(Y).ravel()


def loglikelihood(X, mu, C):
    return logpdf_GAU_ND_Opt(X, mu, C).sum()


def likelihood(X, mu, C):
    Y = np.exp(logpdf_GAU_ND_Opt(X, mu, C))
    return Y.prod()


def pfdND(X, mu, C):
    return np.exp(logpdf_GAU_ND_Opt(X, mu, C))
