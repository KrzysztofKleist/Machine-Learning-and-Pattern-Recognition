from functions.load_functions import *
from functions.plot_functions import *

if __name__ == '__main__':
    D, L = load('files/Train.txt')  # loads the data
    plot_hist(D, L, 'preprocessed_data')  # plots histograms for preprocessed data

    Dg = gaussianization(D)  # gaussianizes the data, same shape as for preprocessed data
    D = Dg
    plot_hist(Dg, L, 'gaussianized_data')  # plots histograms for gaussianized data

    heat_maps(Dg, L)  # plots heat maps for gaussianized data
    plt.show()