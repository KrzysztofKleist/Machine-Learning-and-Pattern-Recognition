import csv
import time

from classifiers import *
from k_folds import *
from pca import *

if __name__ == '__main__':
    print("#####################################################################################")

    # remember - don't use LDA as dim reduction, only PCA
    # we can use LDA as classifier though but it's not good

    D, L = load('files/Train.txt')
    # plot_hist(D, L)
    # plot_scatter(D, L)

    # pca(D, L, 2)
    # lda(D, L, 2)

    # open the file in the write mode
    with open('results.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        start = time.time()

        print("Multivariate Gaussian Classifier")
        classifierType = "Multivariate Gaussian Classifier "
        for k in range(2, D.shape[1]):
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                LTestConcat = np.concatenate((LTestConcat, mulGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

            acc = (L == LTestConcat).sum() / L.size
            err = (1 - acc)
            row = [classifierType + "k = " + str(k), str(acc), str(err)]
            writer.writerow(row)

        end = time.time()
        print("Time elapsed: ", round(end - start, 2), " s")

        print("#####################################################################################")

        print("Naive Bayes Gaussian Classifier")
        classifierType = "Naive Bayes Gaussian Classifier "
        for k in range(2, D.shape[1], 100):
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                LTestConcat = np.concatenate((LTestConcat, naiBayGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

            acc = (L == LTestConcat).sum() / L.size
            err = (1 - acc)
            row = [classifierType + "k = " + str(k), str(acc), str(err)]
            writer.writerow(row)

        print("#####################################################################################")

        print("Tied Multivariate Gaussian Classifier")
        classifierType = "Tied Multivariate Gaussian Classifier "
        for k in range(2, D.shape[1], 100):
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                LTestConcat = np.concatenate((LTestConcat, tiedMulGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

            acc = (L == LTestConcat).sum() / L.size
            err = (1 - acc)
            row = [classifierType + "k = " + str(k), str(acc), str(err)]
            writer.writerow(row)

        print("#####################################################################################")

        print("Tied Naive Bayes Gaussian Classifier")
        classifierType = "Tied Naive Bayes Gaussian Classifier "
        for k in range(2, D.shape[1], 100):
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                LTestConcat = np.concatenate((LTestConcat, tiedNaiBayGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

            acc = (L == LTestConcat).sum() / L.size
            err = (1 - acc)
            row = [classifierType + "k = " + str(k), str(acc), str(err)]
            writer.writerow(row)

        print("#####################################################################################")

        print("Multivariate Gaussian Classifier + PCA")
        classifierType = "Multivariate Gaussian Classifier + PCA "
        for m in range(2, 11):
            print("#################")
            print("PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in range(2, D.shape[1], 100):
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                    LTestConcat = np.concatenate((LTestConcat, mulGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

                acc = (L == LTestConcat).sum() / L.size
                err = (1 - acc)
                row = [classifierType + "k = " + str(k) + " m = " + str(m), str(acc), str(err)]
                writer.writerow(row)

        print("#####################################################################################")

        print("Naive Bayes Gaussian Classifier + PCA")
        classifierType = "Naive Bayes Gaussian Classifier + PCA "
        for m in range(2, 11):
            print("#################")
            print("PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in range(2, D.shape[1], 100):
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                    LTestConcat = np.concatenate((LTestConcat, naiBayGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

                acc = (L == LTestConcat).sum() / L.size
                err = (1 - acc)
                row = [classifierType + "k = " + str(k) + " m = " + str(m), str(acc), str(err)]
                writer.writerow(row)

        print("#####################################################################################")

        print("Tied Multivariate Gaussian Classifier + PCA")
        classifierType = "Tied Multivariate Gaussian Classifier + PCA "
        for m in range(2, 11):
            print("#################")
            print("PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in range(2, D.shape[1], 100):
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                    LTestConcat = np.concatenate((LTestConcat, tiedMulGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

                acc = (L == LTestConcat).sum() / L.size
                err = (1 - acc)
                row = [classifierType + "k = " + str(k) + " m = " + str(m), str(acc), str(err)]
                writer.writerow(row)

        print("#####################################################################################")

        print("Tied Naive Bayes Gaussian Classifier + PCA")
        classifierType = "Tied Naive Bayes Gaussian Classifier + PCA "
        for m in range(2, 11):
            print("#################")
            print("PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in range(2, D.shape[1], 100):
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                    LTestConcat = np.concatenate((LTestConcat, tiedNaiBayGaussClass(DTrain, DTest, LTrain, LTest)),
                                                 axis=0)

                acc = (L == LTestConcat).sum() / L.size
                err = (1 - acc)
                row = [classifierType + "k = " + str(k) + " m = " + str(m), str(acc), str(err)]
                writer.writerow(row)

        print("#####################################################################################")
