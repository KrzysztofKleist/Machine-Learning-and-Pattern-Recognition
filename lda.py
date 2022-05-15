import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy
import scipy.linalg

from loads import *
from plots import *


def lda(D, L):
    mu = mcol(D.mean(1))  # gives the same result as loop and division but faster

    mu0 = mcol(D[:, L == 0].mean(1))
    mu1 = mcol(D[:, L == 1].mean(1))
    muc = np.concatenate((mu0, mu1), axis=1)
    muc = muc - mu
    Sb = np.dot(muc, muc.T)
    SB = Sb / float(muc.shape[1])

    Sw0 = D[:, L == 0] - mu0
    Sw1 = D[:, L == 1] - mu1
    Sw = np.concatenate((Sw0, Sw1), axis=1)
    SW = np.dot(Sw, Sw.T) / float(Sw.shape[1])

    s, U = scipy.linalg.eigh(SB, SW)
    m = 2
    W = U[:, ::-1][:, 0:m]

    WP = np.dot(W.T, D)

    plt.scatter(WP[0][:], WP[1][:])
    plt.show()
