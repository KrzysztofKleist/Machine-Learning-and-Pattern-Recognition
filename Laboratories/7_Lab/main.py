import numpy as np
import numpy.linalg
import scipy.optimize
import sklearn.datasets
from matplotlib import pylab


def mcol(v):
    return v.reshape((v.size, 1))


def func(x):
    y = x[0]
    z = x[1]
    res = np.square(y + 3) + np.sin(y) + np.square(z + 1)
    print(res)
    return res


def funcWithGrad(x):
    y = x[0]
    z = x[1]
    res = np.square(y + 3) + np.sin(y) + np.square(z + 1)
    resGrad1 = 2 * (y + 3) + np.cos(y)
    resGrad2 = 2 * (z + 1)
    resArr = np.array([resGrad1, resGrad2])
    return res, resArr


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def compute_conf_matrix_binary(Pred, Labels):
    C = np.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (Labels == 1)).sum()
    return C


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    mu = numpy.mean(scores)
    if th is None:
        th = -np.log(pi * Cfn) + np.log((1 - pi) * Cfp)
    # P = scores > th * mu
    P = scores > th
    return np.int32(P)


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi * Cfn, (1 - pi) * Cfp)


def bayes_error_plot(pArray, scores, labels, minCost=False):
    y = []
    for p in pArray:
        pi = 1.0 / (1.0 + np.exp(-p))
        # print(pi)
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return np.array(y)


class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.Z = LTR * 2.0 - 1.0
        self.M = DTR.shape[0]

    def logreg_obj(self, v):
        w = v[0:self.M]
        b = v[-1]
        S = numpy.dot(w.T, self.DTR)
        S = S + b
        crossEntropy = numpy.logaddexp(0, -S * self.Z).mean()
        return crossEntropy + 0.5 * self.l * numpy.linalg.norm(w) ** 2


if __name__ == '__main__':
    # functions
    """
    x1, f1, d1 = scipy.optimize.fmin_l_bfgs_b(func,
                                              x0=np.array([0, 0]),
                                              approx_grad=True)
    print("#########################################")
    print("x1 = ", x1, "\nf1 = ", f1, "\nd1 = ", d1)
    
    x2, f2, d2 = scipy.optimize.fmin_l_bfgs_b(funcWithGrad,
                                              x0=np.array([0, 0]))
    print("#########################################")
    print("x2 = ", x2, "\nf2 = ", f2, "\nd2 = ", d2)

    print("#########################################")
    """
    # proper program

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    x0 = numpy.zeros(DTR.shape[0] + 1)
    # for l in [1e-6, 1e-3, 0.1, 1.0]:
    for l in [1e-6]:
        logRegObj = logRegClass(DTR, LTR, l)
        LG_obj = logRegObj.logreg_obj
        v, J, d = scipy.optimize.fmin_l_bfgs_b(LG_obj, x0, approx_grad=True)
        w = v[0:-1]
        b = v[-1]
        STE = numpy.dot(w.T, DTE) + b
        predictedLabels = STE > 0
        wrongPredictions = (LTE != predictedLabels).sum()
        samplesNumber = DTE.shape[1]
        errorRate = float(wrongPredictions / samplesNumber * 100)
        # print(compute_conf_matrix_binary(predictedLabels, LTE))
        Scores = STE
        # pi = 0.9

        # print(Scores - numpy.log(pi/(1-pi)))
        # print(compute_conf_matrix_binary(assign_labels(Scores, pi, 1, 1), LTE))

        print(compute_min_DCF(Scores, LTE, 0.5, 1, 1))
        print(compute_min_DCF(Scores, LTE, 0.1, 1, 1))
        print(compute_min_DCF(Scores, LTE, 0.9, 1, 1))
        print()
        # p = np.linspace(-1, 1, 21)
        # pylab.figure()
        # pylab.plot(p, bayes_error_plot(p, STE, LTE, minCost=False), color='r')
        # pylab.plot(p, bayes_error_plot(p, STE, LTE, minCost=True), color='b')
        # pylab.show()
