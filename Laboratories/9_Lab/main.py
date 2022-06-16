import sys

import numpy as np
import scipy


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2tol(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    # Training Data
    DTR = D[:, idxTrain]
    # Evaluation Data
    DTE = D[:, idxTest]
    # Training Labels
    LTR = L[idxTrain]
    # Evaluation Labels
    LTE = L[idxTest]

    return [(DTR, LTR), (DTE, LTE)]


def train_SVM_Linear(DTR, LTR, C, K=1):
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))])
    # DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = np.dot(DTREXT.T, DTREXT)
    # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T,DTR)
    # H = numpy.exp(-Dist)

    H = mcol(Z) * mrow(Z) * H

    def JDual(Alpha):
        Ha = np.dot(H, mcol(Alpha))
        aHa = np.dot(mrow(Alpha), Ha)
        a1 = Alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(Alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = np.dot(mrow(w), DTREXT)
        loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * np.linalg.norm(w) ** 2 + C * loss

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=0.0,
        maxiter=100000,
        maxfun=100000
    )

    # print(_x)
    # print(_y)

    wStar = np.dot(DTREXT, mcol(alphaStar) * mcol(Z))

    print(JPrimal(wStar))
    print(JDual(alphaStar)[0][0])
    print(JPrimal(wStar) - JDual(alphaStar)[0][0])


def train_SVM_Kernel(DTR, LTR, C, gamma, K=1):
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    Dist = mcol((DTR ** 2).sum(0)) + mrow((DTR ** 2).sum(0)) - 2 * np.dot(DTR.T, DTR)
    H = np.exp(-gamma * Dist) + K
    H = mcol(Z) * mrow(Z) * H

    def JDual(Alpha):
        Ha = np.dot(H, mcol(Alpha))
        aHa = np.dot(mrow(Alpha), Ha)
        a1 = Alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(Alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=0.0,
        maxiter=100000,
        maxfun=100000
    )

    # print(_x)
    # print(_y)

    print(JDual(alphaStar)[0][0])


if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D, L)

    train_SVM_Linear(DTR, LTR, 1.0, 1.0)
    train_SVM_Kernel(DTR, LTR, 1.0, 1.0)
    sys.exit(0)
