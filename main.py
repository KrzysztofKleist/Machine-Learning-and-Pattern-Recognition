import numpy
import matplotlib
import matplotlib.pyplot as plt

from loads import *
from plots import *
from pca import *
from lda import *

if __name__ == '__main__':
    # Change default font size - comment to use default values
    # plt.rc('font', size=16)
    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)

    D, L = load('files/Train.txt')
    # plot_hist(D, L)
    # plot_scatter(D, L)

    pca(D, L)
    lda(D, L)

