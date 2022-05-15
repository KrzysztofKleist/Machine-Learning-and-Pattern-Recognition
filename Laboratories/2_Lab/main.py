import numpy
import matplotlib
import matplotlib.pyplot as plt

from loads import *
from plots import *


if __name__ == '__main__':

    # Change default font size - comment to use default values
    # plt.rc('font', size=16)
    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)

    D, L = load('iris.csv')
    # plot_hist(D, L)
    # plot_scatter(D, L)


