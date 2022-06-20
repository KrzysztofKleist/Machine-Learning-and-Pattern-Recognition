import sys

import scipy

from cost_computations_functions import *


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

    print(JPrimal(wStar))
    print(JDual(alphaStar)[0][0])
    print(JPrimal(wStar) - JDual(alphaStar)[0][0])
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

    print(JDual(alphaStar)[0][0])
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

    print(JDual(alphaStar)[0][0])
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


if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D, L)

    wStar = train_SVM_Linear(DTR, LTR, 1.0, 1.0)
    w = wStar[0:-1]
    b = wStar[-1]

    scores = np.dot(w.T, DTE) + b
    Pred = scores > 0
    print(((1 - (Pred == LTE).sum() / LTE.shape[0]) * 100).round(1))

    scores = scores.reshape(-1)

    plot_ROC(scores, LTE)
    plt.show()

    # Compute the confusion matrix with the llr calculated previously and with real_labels from the k fold
    prior_tilde = 0.5
    CFN = 1
    CFP = 1

    confusion_matrix = compute_conf_matrix_binary(Pred, LTE)
    print(confusion_matrix)

    # Bayes empirical risk
    Bayes_emp_risk = compute_emp_Bayes_binary(confusion_matrix, prior_tilde, CFN, CFP)
    print(Bayes_emp_risk)

    # Bayes empirical risk with a dummy strategy
    Bayes_emp_risk_dummy = min(prior_tilde * CFN, (1 - prior_tilde) * CFP)
    print(Bayes_emp_risk_dummy)

    # Normalized empirical Bayes risk, actual DCF
    print(compute_act_DCF(scores, LTE, prior_tilde, CFN, CFP))

    # Compute the minimum normalized DCF for our model
    print(compute_min_norm_DCF(scores, LTE, prior_tilde, CFN, CFP))

    bayes_error_plot(np.linspace(-3, 3, 21), scores, LTE)  # np.linspace(-3, 3, 21) - effective prior logOdds
    plt.show()

    #
    #
    #
    #
    # alphaS = train_SVM_Kernel_Poly(DTR, LTR, 1, 2, 1, 1)
    # scores = compute_scores_Poly(alphaS, DTR, LTR, DTE, 2, 1, 1)
    # Pred = scores > 0
    # print(((1 - (Pred == LTE).sum() / LTE.shape[0]) * 100).round(1))

    # alphaS = train_SVM_Kernel_RBF(DTR, LTR, 1.0, 1.0, 0)
    # scores = compute_scores_RBF(alphaS, DTR, LTR, DTE, 1, 0)
    # Pred = scores > 0
    # print(((1 - (Pred == LTE).sum() / LTE.shape[0]) * 100).round(1))

    sys.exit(0)
