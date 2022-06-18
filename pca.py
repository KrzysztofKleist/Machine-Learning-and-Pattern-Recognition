from functions.load_functions import *
from functions.pca_functions import *
from functions.plot_functions import *

if __name__ == '__main__':
    D, L = load('files/Train.txt')  # loads the data
    DP = pca(D, L, 2)
    plot_pca(DP, L)
    plt.savefig('plots/pca.jpg')
    plt.show()

    DG = gaussianization(D)
    DGP = pca(DG, L, 2)
    plot_pca(DGP, L)
    plt.savefig('plots/pca_gaussianized.jpg')
    plt.show()
