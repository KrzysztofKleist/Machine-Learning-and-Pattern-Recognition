import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy
import scipy.linalg

from matrix_transformations import *
from loads import *
from plots import *

if __name__ == '__main__':

    D, L = load('iris.csv')

    mu = mcol(D.mean(1))  # gives the same result as loop and division but faster

    mu0 = mcol(D[:, L == 0].mean(1))
    mu1 = mcol(D[:, L == 1].mean(1))
    mu2 = mcol(D[:, L == 2].mean(1))
    muc = np.concatenate((mu0, mu1, mu2), axis=1)
    muc = muc - mu
    Sb = np.dot(muc, muc.T)
    SB = Sb / float(muc.shape[1])
    # print(SB)
    # print()

    Sw0 = D[:, L == 0] - mu0
    Sw1 = D[:, L == 1] - mu1
    Sw2 = D[:, L == 2] - mu2
    Sw = np.concatenate((Sw0, Sw1, Sw2), axis=1)
    SW = np.dot(Sw, Sw.T) / float(Sw.shape[1])
    # print(SW)
    # print()

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:2]
    print(W)
    # UW, _, _ = numpy.linalg.svd(W)
    # U = UW[:, 0:4]
    # print(U)
    print(np.load("IRIS_LDA_matrix_m2.npy"))
    print(W.round(9) == np.load("IRIS_LDA_matrix_m2.npy").round(9))

    WP = np.dot(W.T, D)

    WP0 = WP[:, L == 0]
    WP1 = WP[:, L == 1]
    WP2 = WP[:, L == 2]

    print(WP0.shape)

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.scatter(WP0[0], WP0[1], c='blue')
    plt.scatter(WP1[0], WP1[1], c='orange')
    plt.scatter(WP2[0], WP2[1], c='green')
    plt.show()
