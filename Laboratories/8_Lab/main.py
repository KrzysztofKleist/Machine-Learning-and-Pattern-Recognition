import numpy.linalg.linalg
import pylab
import scipy.special
import sys

from lab4_functions import *
from loads import *


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


def ML_GAU(D):
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / float(D.shape[1])
    return mu, C


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -np.log(pi * Cfn) + np.log((1 - pi) * Cfp)
    P = scores > th
    return np.int32(P)


def compute_conf_matrix_binary(Pred, Labels):
    C = np.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (Labels == 1)).sum()
    return C


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi * Cfn, (1 - pi) * Cfp)


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


if __name__ == '__main__':

    ##################################################################
    # Loading Iris Dataset
    """
    D, L = load2()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTrain, LTrain), (DTest, LTest) = split_db_2to1(D, L)
    """
    ##################################################################
    # Confusion Matrix for Multivariate Gaussian Classifier
    """
    print("Multivariate Gaussian Classifier")
    h = {}

    for lab in [0, 1, 2]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    logSJoint = np.zeros((3, DTest.shape[1]))
    classPriors = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    for lab in [0, 1, 2]:
        mu, C = h[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred2 = Post2.argmax(0)

    acc = (LPred2 == LTest).sum() / LTest.shape[0]
    err = 1 - acc
    print("Accuracy = ", (acc * 100).__round__(2), "%")
    print("Error rate = ", (err * 100).__round__(2), "%")

    K = len(np.unique(LTest))  # Number of classes
    confMatrix = np.zeros((K, K))

    for i in range(len(LTest)):
        confMatrix[LPred2[i]][LTest[i]] += 1

    print(confMatrix)
    """
    ##################################################################
    # Confusion Matrix for Tied Multivariate Gaussian Classifier
    """
    print("Tied Multivariate Gaussian Classifier")
    h = {}

    for lab in [0, 1, 2]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    Cglobal = np.zeros(C.shape)

    for lab in [0, 1, 2]:
        _, C = h[lab]
        Cglobal = Cglobal + ((LTrain == lab).sum()) * C

    Cglobal = Cglobal / DTrain.shape[1]

    for lab in [0, 1, 2]:
        mu, _ = h[lab]
        h[lab] = (mu, Cglobal)

    logSJoint = np.zeros((3, DTest.shape[1]))
    classPriors = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    for lab in [0, 1, 2]:
        mu, C = h[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred2 = Post2.argmax(0)

    acc = (LPred2 == LTest).sum() / LTest.shape[0]
    err = 1 - acc
    print("Accuracy = ", (acc * 100).__round__(2), "%")
    print("Error rate = ", (err * 100).__round__(2), "%")

    K = len(np.unique(LTest))  # Number of classes
    confMatrix = np.zeros((K, K))

    for i in range(len(LTest)):
        confMatrix[LPred2[i]][LTest[i]] += 1

    print(confMatrix)
    """
    ##################################################################
    # Confusion Matrix for Commedia Dataset
    """
    commedia_labels = numpy.load('Data/commedia_labels.npy')
    commedia_predictedLabels = numpy.load('Data/commedia_predictedLabels.npy')

    acc = (commedia_predictedLabels == commedia_labels).sum() / commedia_labels.shape[0]
    err = 1 - acc
    print("Accuracy = ", (acc * 100).__round__(2), "%")
    print("Error rate = ", (err * 100).__round__(2), "%")

    K = len(np.unique(commedia_labels))  # Number of classes
    confMatrix = np.zeros((K, K))

    for i in range(len(commedia_labels)):
        confMatrix[commedia_predictedLabels[i]][commedia_labels[i]] += 1

    print(confMatrix)
    """
    ##################################################################
    # Confusion Matrix for Commedia Dataset by Cumani

    llCond = np.load('Data/commedia_ll.npy')
    llJoint = llCond + np.log(1.0 / 3.0)
    Labels = np.load('Data/commedia_labels.npy')
    llMarginal = scipy.special.logsumexp(llJoint, axis=0)
    Post = np.exp(llJoint - llMarginal)
    Pred = np.argmax(Post, axis=0)
    print("Post shape: ", Post.shape)
    print("Pred shape: ", Pred.shape)
    print("Labels shape: ", Labels.shape)
    # Conf = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         Conf[i, j] = ((Pred == i) * (Labels == j)).sum()
    #
    # print(Conf)
    sys.exit(0)

    ##################################################################
    # Binary task: optimal decisions
    """
    llrs = np.load('Data/commedia_llr_infpar.npy')
    Labels = np.load('Data/commedia_labels_infpar.npy')

    thresholds = np.array(llrs)
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(llrs > t)
        Conf = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Conf[i, j] = ((Pred == i) * (Labels == j)).sum()
        TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1])
        FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])

    pylab.figure()
    pylab.plot(FPR, TPR)
    # pylab.show()

    scores = llrs

    print(compute_act_DCF(scores, Labels, 0.8, 1, 10))
    print(compute_min_DCF(scores, Labels, 0.8, 1, 10))

    p = np.linspace(-3, 3, 21)
    pylab.figure()
    pylab.plot(p, bayes_error_plot(p, scores, Labels, minCost=False), color='r')
    pylab.plot(p, bayes_error_plot(p, scores, Labels, minCost=True), color='b')
    pylab.show()
    """