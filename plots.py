import matplotlib
import matplotlib.pyplot as plt


def plot_hist(D, L):
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
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='Bad')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='Good')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        # plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()


def plot_scatter(D, L):
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

    for dIdx1 in range(11):
        for dIdx2 in range(11):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Bad')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Good')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            # plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()
