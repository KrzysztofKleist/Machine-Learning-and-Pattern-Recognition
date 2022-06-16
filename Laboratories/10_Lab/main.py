import json

import numpy as np
import numpy.linalg.linalg
import scipy
import scipy.special


# import scipy.linalg
# import sys
# import sklearn.datasets

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def logpdf_GAU_ND_Opt(X, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * X.shape[0] * np.log(2 * np.pi)
    const += -0.5 * np.linalg.slogdet(C)[1]

    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i + 1]
        res = const + -0.5 * np.dot((x - mu).T, np.dot(P, (x - mu)))
        Y.append(res)

    return np.array(Y).ravel()

    Y = [logpdf_1sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


def logpdf_1sample(x, mu, C):
    P = np.linalg.inv(C)
    res = -0.5 * x.shape[0] * np.log(2 * np.pi)
    res += -0.5 * np.linalg.slogdet(C)[1]
    res += -0.5 * np.dot((x - mu).T, np.dot(P, (x - mu)))
    return res.ravel()


def GMM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)


def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        P = np.exp(SJ - SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        print(llNew)
    print(llNew - llOld)
    return gmm


if __name__ == '__main__':
    gmm = load_gmm('Data/GMM_1D_3G_EM.json')
    print(gmm)
