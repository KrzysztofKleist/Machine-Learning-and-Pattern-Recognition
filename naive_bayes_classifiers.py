from functions.cost_computations_functions import *
from functions.gaussian_classifiers_functions import *
from functions.kfolds_functions import *
from functions.load_functions import *
from functions.pca_functions import *

if __name__ == '__main__':
    D, L = load('files/Train.txt')  # loads the data

    # comment next two lines to get raw features
    # Dg = gaussianization(D)  # gaussianizes the data, same shape as for preprocessed data
    # D = Dg

    m = 8
    D = pca(D, L, m)

    kRange = [5]
    # pcaMRange = [8]
    pcaMRange = [8, 9]

    print("#####################################################################################")
    print("Naive Bayes Gaussian Classifier")

    classifierType = "Naive Bayes Gaussian Classifier "
    for k in kRange:
        print("k: {}".format(k), "\r", end="")
        LTestConcat = np.empty([0])
        ScoresConcat = np.empty([0])
        for iter in range(k):
            DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
            LPred2, scores = naiBayGaussClass(DTrain, DTest, LTrain, LTest)
            LTestConcat = np.concatenate((LTestConcat, LPred2), axis=0)
            ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.5, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.5, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.1, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.1, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.9, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.9, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

    print("#####################################################################################")
    print("Tied Naive Bayes Gaussian Classifier")

    classifierType = "Tied Naive Bayes Gaussian Classifier "
    for k in kRange:
        print("k: {}".format(k), "\r", end="")
        LTestConcat = np.empty([0])
        ScoresConcat = np.empty([0])
        for iter in range(k):
            DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
            LPred2, scores = tiedNaiBayGaussClass(DTrain, DTest, LTrain, LTest)
            LTestConcat = np.concatenate((LTestConcat, LPred2), axis=0)
            ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.5, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.5, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.1, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.1, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

        CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.9, 1, 1), L)
        print(CM)
        print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.9, 1, 1).round(3))
        print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

    ###### THIS CODE IS USELESS ######
    # print("#####################################################################################")
    # print("Naive Bayes Gaussian Classifier + PCA")
    """
    classifierType = "Naive Bayes Gaussian Classifier + PCA "
    for m in pcaMRange:
        print(" #################")
        print(" PCA for m = " + str(m))
        DP = pca(D, L, m)
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                LPred2, scores = naiBayGaussClass(DTrain, DTest, LTrain, LTest)
                LTestConcat = np.concatenate((LTestConcat, LPred2), axis=0)
                ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.5, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.5, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.1, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.1, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.9, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.9, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")
    """
    # print("#####################################################################################")
    # print("Tied Naive Bayes Gaussian Classifier + PCA")
    """
    classifierType = "Tied Naive Bayes Gaussian Classifier + PCA "
    for m in pcaMRange:
        print(" #################")
        print(" PCA for m = " + str(m))
        DP = pca(D, L, m)
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                LPred2, scores = tiedNaiBayGaussClass(DTrain, DTest, LTrain, LTest)
                LTestConcat = np.concatenate((LTestConcat, LPred2), axis=0)
                ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.5, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.5, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.1, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.1, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")

            CM = compute_conf_matrix_binary(assign_labels(ScoresConcat, 0.9, 1, 1), L)
            print(CM)
            print("minDCF: ", compute_min_DCF(ScoresConcat, L, 0.9, 1, 1).round(3))
            print("error rate: ", compute_error_rate(CM).round(4) * 100, " %")
    """
