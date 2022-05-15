import numpy
import matplotlib
import matplotlib.pyplot as plt

from loads import *
from plots import *


def pca(D, L):
    mu = mcol(D.mean(1))

    DC = D - mcol(mu)

    C = np.dot(DC, DC.T)  # covariance matrix
    C = C / float(D.shape[1])

    s, U = np.linalg.eigh(C)

    m = 2
    P = U[:, ::-1][:, 0:m]

    DP = np.dot(P.T, D)

    plt.scatter(DP[0][:], DP[1][:])
    plt.show()
