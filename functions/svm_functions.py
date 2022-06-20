import scipy

from .matrix_transformations_functions import *


def train_SVM_Linear(DTR, LTR, C, K=1):
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])

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

    # print(JPrimal(wStar))
    # print(JDual(alphaStar)[0][0])
    # print(JPrimal(wStar) - JDual(alphaStar)[0][0])
    return wStar


def train_SVM_Kernel_Poly(DTR, LTR, C, d, c, K=1):
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    kernel = (np.dot(DTR.T, DTR) + c) ** d + K
    H = mcol(Z) * mrow(Z) * kernel

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

    # print(JDual(alphaStar)[0][0])
    return alphaStar


def compute_scores_Poly(alphaStar, DTR, LTR, DTE, d, c, K):
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    scores = np.zeros(DTE.shape[1])
    for j in range(DTE.shape[1]):
        for i in range(DTR.shape[1]):
            scores[j] = scores[j] + alphaStar[i] * Z[i] * (((np.dot(DTR[:, i].T, DTE[:, j]) + c) ** d) + K)
    return scores


def train_SVM_Kernel_RBF(DTR, LTR, C, gamma, K=1):
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

    # print(JDual(alphaStar)[0][0])
    return alphaStar


def compute_scores_RBF(alphaStar, DTR, LTR, DTE, gamma, K):
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    scores = np.zeros(DTE.shape[1])
    for j in range(DTE.shape[1]):
        for i in range(DTR.shape[1]):
            scores[j] = scores[j] + alphaStar[i] * Z[i] * (
                    np.exp(- gamma * np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2) + K)
    return scores
