import csv
import time

import scipy.optimize

from classifiers import *
from k_folds import *
from pca import *


class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.Z = LTR * 2.0 - 1.0
        self.M = DTR.shape[0]

    def logreg_obj(self, v):
        w = v[0:self.M]
        b = v[-1]
        S = numpy.dot(w.T, self.DTR)
        S = S + b
        crossEntropy = numpy.logaddexp(0, -S * self.Z).mean()
        return crossEntropy + 0.5 * self.l * numpy.linalg.norm(w) ** 2


if __name__ == '__main__':
    print("#####################################################################################")
    print("####################################### START #######################################")
    start = time.time()

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

        # kRange = [3, 10, 46, 47, 49, 57, 62, 63, 103, 108, 613, 919,
        #           1839]  # values of k are chosen to get the most even ranges of kFold output
        # kRange = [3, 10, 50, 100, 613, 1839]
        kRange = [3, 5, 10]
        pcaMRange = [7, 8, 9, 10]

        print("#####################################################################################")
        print("Multivariate Gaussian Classifier")
        classifierType = "Multivariate Gaussian Classifier "
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                LTestConcat = np.concatenate((LTestConcat, mulGaussClass(DTrain, DTest, LTrain, LTest)), axis=0)

            acc = (L == LTestConcat).sum() / L.size
            err = (1 - acc)
            row = [classifierType + "k = " + str(k), str(acc), str(err)]
            writer.writerow(row)

        print("#####################################################################################")
        print("Naive Bayes Gaussian Classifier")
        classifierType = "Naive Bayes Gaussian Classifier "
        for k in kRange:
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
        for k in kRange:
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
        for k in kRange:
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
        for m in pcaMRange:
            print(" #################")
            print(" PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in kRange:
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
        for m in pcaMRange:
            print(" #################")
            print(" PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in kRange:
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
        for m in pcaMRange:
            print(" #################")
            print(" PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in kRange:
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
        for m in pcaMRange:
            print(" #################")
            print(" PCA for m = " + str(m))
            DP = pca(D, L, m)
            for k in kRange:
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
        print("Logistic Regression")

        lRange = [1e-6, 1e-3, 0.1, 1.0]
        # kRange = [3, 10, 50, 100]

        classifierType = "Logistic Regression "
        for l in lRange:
            print("     ########")
            print("     for l = " + str(l))
            for k in kRange:
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                    x0 = numpy.zeros(DTrain.shape[0] + 1)
                    logRegObj = logRegClass(DTrain, LTrain, l)
                    v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
                    w = v[0:-1]
                    b = v[-1]
                    STE = numpy.dot(w.T, DTest) + b
                    predictedLabels = STE > 0
                    LTestConcat = np.concatenate((LTestConcat, predictedLabels), axis=0)

                acc = (L == LTestConcat).sum() / L.size
                err = (1 - acc)
                row = [classifierType + "l = " + str(l) + " k = " + str(k), str(acc), str(err)]
                writer.writerow(row)

        print("#####################################################################################")
        print("Logistic Regression + PCA")

        classifierType = "Logistic Regression "
        for m in pcaMRange:
            print(" #################")
            print(" PCA for m = " + str(m))
            DP = pca(D, L, m)
            for l in lRange:
                print("     ########")
                print("     for l = " + str(l))
                for k in kRange:
                    print("k: {}".format(k), "\r", end="")
                    LTestConcat = np.empty([0])
                    for iter in range(k):
                        DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                        x0 = numpy.zeros(DTrain.shape[0] + 1)
                        logRegObj = logRegClass(DTrain, LTrain, l)
                        v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
                        w = v[0:-1]
                        b = v[-1]
                        STE = numpy.dot(w.T, DTest) + b
                        predictedLabels = STE > 0
                        LTestConcat = np.concatenate((LTestConcat, predictedLabels), axis=0)

                    acc = (L == LTestConcat).sum() / L.size
                    err = (1 - acc)
                    row = [classifierType + "l = " + str(l) + " k = " + str(k) + " m = " + str(m), str(acc), str(err)]
                    writer.writerow(row)

        print("#####################################################################################")
        end = time.time()
        print("Time elapsed: ", round(end - start, 2), " s")
        print("#####################################################################################")
        print("######################################## END ########################################")
        print("#####################################################################################")
