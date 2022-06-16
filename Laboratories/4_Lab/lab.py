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


plt.figure()
XPlot = numpy.linspace(-8, 12, 1000)
m = numpy.ones((1, 1)) * 1.0
C = numpy.ones((1, 1)) * 2.0
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
plt.show()

pdfSol = numpy.load('solutions/llGAU.npy')
pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
print(numpy.abs(pdfSol - pdfGau).max())

XND = numpy.load('solutions/XND.npy')
mu = numpy.load('solutions/muND.npy')
C = numpy.load('solutions/CND.npy')
pdfSol = numpy.load('solutions/llND.npy')
pdfGau = logpdf_GAU_ND(XND, mu, C)
print(numpy.abs(pdfSol - pdfGau).max().round(10))

m_ML = compute_empirical_mean(XND)
C_ML = compute_empirical_cov(XND)

ll = loglikelihood(XND, m_ML, C_ML)
print(ll)

X1D = numpy.load('solutions/X1D.npy')
m_ML = compute_empirical_mean(X1D)
C_ML = compute_empirical_cov(X1D)

plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = numpy.linspace(-8, 12, 1000)
plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
plt.show()

ll = loglikelihood(X1D, m_ML, C_ML)
# print(ll)


