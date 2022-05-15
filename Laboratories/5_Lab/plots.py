import matplotlib.pyplot as plt


def plot_hist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='Setosa')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='Versicolor')
        plt.hist(D2[dIdx, :], bins=10, density=True, alpha=0.4, label='Virginica')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        # plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()


def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label='Virginica')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            # plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()
