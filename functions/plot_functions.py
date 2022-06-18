import matplotlib.pyplot as plt

from .matrix_transformations_functions import *


def plot_hist(D, L, name):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
    }

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins=50, density=True, alpha=0.7, label='Bad')
        plt.hist(D1[dIdx, :], bins=50, density=True, alpha=0.7, label='Good')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        # plt.savefig('plots/' + name + '_' + str(dIdx) + '_hist_' + hFea[dIdx].replace(" ", "_") + '.jpg')
    # plt.show()


def plot_pca(DP, L):
    DP0 = DP[:, L == 0]
    DP1 = DP[:, L == 1]

    plt.figure()
    plt.scatter(DP0[0], DP0[1], c='blue', label='Bad')
    plt.scatter(DP1[0], DP1[1], c='orange', label='Good')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('plots/pca.jpg')
    # plt.show()


def heat_maps(Dg, L):
    Dh = abs(compute_empirical_cov(Dg))
    plt.figure()
    plt.title('all wines')
    plt.imshow(Dh, cmap='Greys', aspect='equal')
    # plt.savefig('plots/heat_map_0_all_wines.jpg')

    DhGood = abs(compute_empirical_cov(Dg[:, L == 1]))
    plt.figure()
    plt.title('good wines')
    plt.imshow(DhGood, cmap='Reds', aspect='equal')
    # plt.savefig('plots/heat_map_1_good_wines.jpg')

    DhBad = abs(compute_empirical_cov(Dg[:, L == 0]))
    plt.figure()
    plt.title('bad wines')
    plt.imshow(DhBad, cmap='Blues', aspect='equal')
    # plt.savefig('plots/heat_map_2_bad_wines.jpg')
    # plt.show()
