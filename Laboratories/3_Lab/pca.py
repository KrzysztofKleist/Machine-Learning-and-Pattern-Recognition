import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy

from matrix_transformations import *
from loads import *
from plots import *

if __name__ == '__main__':

    D, L = load('iris.csv')
    # print(D)
    # print("\n\n\n")

    mu = mcol(D.mean(1))  # gives the same result as loop and division but faster

    DC = D - mcol(mu)

    C = np.dot(DC, DC.T)     # covariance matrix
    C = C / float(D.shape[1])

    s, U = np.linalg.eigh(C)

    P = U[:, ::-1][:, 0:2]

    print(P)
    print(np.load("IRIS_PCA_matrix_m4.npy"))
    print(P.round(9) == np.load("IRIS_PCA_matrix_m4.npy").round(9))

    DP = np.dot(P.T, D)

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.scatter(DP[0][0:49], DP[1][0:49], c='blue')
    plt.scatter(DP[0][50:99], DP[1][50:99], c='orange')
    plt.scatter(DP[0][100:], DP[1][100:], c='green')
    plt.show()
