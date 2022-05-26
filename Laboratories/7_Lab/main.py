import numpy as np
import numpy.linalg
import scipy.optimize
import sklearn.datasets


def mcol(v):
    return v.reshape((v.size, 1))


def func(x):
    y = x[0]
    z = x[1]
    res = np.square(y + 3) + np.sin(y) + np.square(z + 1)
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
    """
    # functions
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

    for l in [1e-6, 1e-3, 0.1, 1.0]:
        logRegObj = logRegClass(DTR, LTR, l)
        v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
        w = v[0:-1]
        b = v[-1]
        STE = numpy.dot(w.T, DTE) + b
        predictedLabels = STE > 0
        wrongPredictions = (LTE != predictedLabels).sum()
        samplesNumber = DTE.shape[1]
        errorRate = float(wrongPredictions / samplesNumber * 100)
        print(errorRate)
