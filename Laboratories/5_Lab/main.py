import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg.linalg
import pylab
import scipy
import scipy.linalg

from matrix_transformations import *
from loads import *
from plots import *
from lab4_functions import *


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


if __name__ == '__main__':
    D, L = load2()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTrain, LTrain), (DTest, LTest) = split_db_2to1(D, L)

    h = {}

    for lab in [0, 1, 2]:
        mu, C = ML_GAU(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)

    SJoint = np.zeros((3, DTest.shape[1]))
    logSJoint = np.zeros((3, DTest.shape[1]))
    classPriors = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    for lab in [0, 1, 2]:
        mu, C = h[lab]
        SJoint[lab, :] = np.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + np.log((classPriors[lab]))

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / SMarginal
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)



    # plt.figure()
    # XPlot = numpy.linspace(-8, 12, 1000)
    # m = numpy.ones((1, 1)) * 1.0
    # C = numpy.ones((1, 1)) * 2.0
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    # # plt.show()
    #
    # pdfSol = numpy.load('solutions4/llGAU.npy')
    # pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    # print(numpy.abs(pdfSol - pdfGau).max())
    #
    # XND = numpy.load('solutions4/XND.npy')
    # mu = numpy.load('solutions4/muND.npy')
    # C = numpy.load('solutions4/CND.npy')
    # pdfSol = numpy.load('solutions4/llND.npy')
    # pdfGau = logpdf_GAU_ND(XND, mu, C)
    # print(numpy.abs(pdfSol - pdfGau).max().round(10))
    #
    # m_ML = compute_empirical_mean(XND)
    # C_ML = compute_empirical_cov(XND)
    #
    # ll = loglikelihood(XND, m_ML, C_ML)
    # print(ll)
    #
    # X1D = numpy.load('solutions4/X1D.npy')
    # m_ML = compute_empirical_mean(X1D)
    # C_ML = compute_empirical_cov(X1D)
    #
    # plt.figure()
    # plt.hist(X1D.ravel(), bins=50, density=True)
    # XPlot = numpy.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
    # plt.show()
    #
    # ll = loglikelihood(X1D, m_ML, C_ML)
    # print(ll)
