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

    DP0 = DP[:, L == 0]
    DP1 = DP[:, L == 1]

    plt.title('PCA')
    plt.scatter(DP0[0][:], DP0[1][:], c='blue', label='Bad wine')
    plt.scatter(DP1[0][:], DP1[1][:], c='orange', label='Good wine')
    plt.legend()
    plt.show()
