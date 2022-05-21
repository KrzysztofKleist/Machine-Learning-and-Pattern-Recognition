import scipy
import scipy.linalg
import scipy.special

from ml_gau import *


def logpdf_1sample(x, mu, C):
    P = np.linalg.inv(C)
    res = -0.5 * x.shape[0] * np.log(2 * np.pi)
    res += -0.5 * np.linalg.slogdet(C)[1]
    res += -0.5 * np.dot((x - mu).T, np.dot(P, (x - mu)))
    return res.ravel()


def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


def mulGaussClass(DTrain, DTest, LTrain, LTest):
    h = {}

    for lab in [0, 1]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    SJoint = np.zeros((2, DTest.shape[1]))
    logSJoint = np.zeros((2, DTest.shape[1]))
    classPriors = [1226.0 / 1839.0, 613.0 / 1839.0]

    for lab in [0, 1]:
        mu, C = h[lab]
        SJoint[lab, :] = np.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    SMarginal = vrow(SJoint.sum(0))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

    Post1 = SJoint / SMarginal
    LPred1 = Post1.argmax(0)

    logPost = logSJoint - logSMarginal
    Post2 = np.exp(logPost)
    LPred2 = Post2.argmax(0)

    return LPred2


def naiBayGaussClass(DTrain, DTest, LTrain, LTest):
    h = {}

    for lab in [0, 1]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        C = C * np.identity(C.shape[0])
        h[lab] = (mu, C)

    SJoint = np.zeros((2, DTest.shape[1]))
    logSJoint = np.zeros((2, DTest.shape[1]))
    classPriors = [1226.0 / 1839.0, 613.0 / 1839.0]

    for lab in [0, 1]:
        mu, C = h[lab]
        SJoint[lab, :] = np.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    SMarginal = vrow(SJoint.sum(0))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

    Post1 = SJoint / SMarginal
    LPred1 = Post1.argmax(0)

    logPost = logSJoint - logSMarginal
    Post2 = np.exp(logPost)
    LPred2 = Post2.argmax(0)

    return LPred2


def tiedMulGaussClass(DTrain, DTest, LTrain, LTest):
    h = {}

    for lab in [0, 1]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    Cglobal = np.zeros(C.shape)

    for lab in [0, 1]:
        _, C = h[lab]
        Cglobal = Cglobal + ((LTrain == lab).sum()) * C

    Cglobal = Cglobal / DTrain.shape[1]

    for lab in [0, 1]:
        mu, _ = h[lab]
        h[lab] = (mu, Cglobal)

    SJoint = np.zeros((2, DTest.shape[1]))
    logSJoint = np.zeros((2, DTest.shape[1]))
    classPriors = [1226.0 / 1839.0, 613.0 / 1839.0]

    for lab in [0, 1]:
        mu, C = h[lab]
        SJoint[lab, :] = np.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    SMarginal = vrow(SJoint.sum(0))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

    Post1 = SJoint / SMarginal
    LPred1 = Post1.argmax(0)

    logPost = logSJoint - logSMarginal
    Post2 = np.exp(logPost)
    LPred2 = Post2.argmax(0)

    return LPred2


def tiedNaiBayGaussClass(DTrain, DTest, LTrain, LTest):
    h = {}

    for lab in [0, 1]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    Cglobal = np.zeros(C.shape)

    for lab in [0, 1]:
        _, C = h[lab]
        Cglobal = Cglobal + ((LTrain == lab).sum()) * C

    Cglobal = Cglobal / DTrain.shape[1]

    for lab in [0, 1]:
        mu, _ = h[lab]
        h[lab] = (mu, Cglobal * np.identity(Cglobal.shape[0]))

    SJoint = np.zeros((2, DTest.shape[1]))
    logSJoint = np.zeros((2, DTest.shape[1]))
    classPriors = [1226.0 / 1839.0, 613.0 / 1839.0]

    for lab in [0, 1]:
        mu, C = h[lab]
        SJoint[lab, :] = np.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    SMarginal = vrow(SJoint.sum(0))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))

    Post1 = SJoint / SMarginal
    LPred1 = Post1.argmax(0)

    logPost = logSJoint - logSMarginal
    Post2 = np.exp(logPost)
    LPred2 = Post2.argmax(0)

    return LPred2