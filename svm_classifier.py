from functions.load_functions import *
from functions.svm_functions import *
from functions.plot_functions import *
from functions.cost_computations_functions import *
from functions.kfolds_functions import *

if __name__ == '__main__':
    D, L = load('files/Train.txt')  # loads the data

    # comment next two lines to get raw features
    Dg = gaussianization(D)  # gaussianizes the data, same shape as for preprocessed data
    D = Dg

    kRange = [5]

    # CRange for minDCF plots
    CRange = np.logspace(-2, 0, 10, endpoint=True, base=10.0)
    # CRange = [0.1]

    prior_T = 0.5  # change between 0.5, 0.1, 0.9

    CArr = np.array(CRange)
    minDCF_0_5 = np.empty([0])
    minDCF_0_1 = np.empty([0])
    minDCF_0_9 = np.empty([0])

    print("#####################################################################################")
    print("SVM Linear Kernel")

    classifierType = "SVM Linear Kernel "
    for C in CRange:
        print("     ########")
        print("     for c = " + str(C))
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                wStar = train_SVM_Linear(DTrain, LTrain, C, 1.0)
                w = wStar[0:-1]
                b = wStar[-1]
                scores = np.dot(w.T, DTest) + b
                scores = scores.reshape(-1)

                ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)
                predictedLabels = scores > 0
                LTestConcat = np.concatenate((LTestConcat, predictedLabels), axis=0)

            minDCF_0_5 = np.concatenate((minDCF_0_5, np.array([compute_min_DCF(ScoresConcat, L, 0.5, 1, 1)])),
                                        axis=0)
            minDCF_0_1 = np.concatenate((minDCF_0_1, np.array([compute_min_DCF(ScoresConcat, L, 0.1, 1, 1)])),
                                        axis=0)
            minDCF_0_9 = np.concatenate((minDCF_0_9, np.array([compute_min_DCF(ScoresConcat, L, 0.9, 1, 1)])),
                                        axis=0)

        print(minDCF_0_5.round(3))
        print(minDCF_0_1.round(3))
        print(minDCF_0_9.round(3))

        bayes_error_plot(np.linspace(-2, 2, 21), ScoresConcat, L)
        plt.savefig('plots/bayes_error_plot_SVM_linear_gaussianized_data.jpg')
        plot_ROC(ScoresConcat, L)
        plt.savefig('plots/roc_plot_SVM_linear_gaussianized_data.jpg')
        # plt.show()


    # # minDCF plots
    # plt.figure()
    # plt.xscale('log')
    # plt.plot(CArr, minDCF_0_5, label='minDCF-0.5', color='r')
    # plt.plot(CArr, minDCF_0_1, label='minDCF-0.1', color='b')
    # plt.plot(CArr, minDCF_0_9, label='minDCF-0.9', color='g')
    #
    # plt.xlabel('C')
    # plt.ylabel("DCF")
    # plt.tight_layout()
    # plt.legend()
    # # plt.savefig('plots/minDCF_SVM_linear_raw_data.jpg')
    # plt.savefig('plots/minDCF_SVM_linear_gaussianized_data.jpg')
    # plt.show()
    # sys.exit(0)

    print("#####################################################################################")
    print("SVM Polynomial Kernel")
    """
    classifierType = "SVM Polynomial Kernel "
    for C in CRange:
        print("     ########")
        print("     for C = " + str(C))
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                alphaS = train_SVM_Kernel_Poly(DTrain, LTrain, C, 2, 1, 1)
                scores = compute_scores_Poly(alphaS, DTrain, LTrain, DTest, 2, 1, 1)

                ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)
                predictedLabels = scores > 0
                LTestConcat = np.concatenate((LTestConcat, predictedLabels), axis=0)

            minDCF_0_5 = np.concatenate((minDCF_0_5, np.array([compute_min_DCF(ScoresConcat, L, 0.5, 1, 1)])),
                                        axis=0)
            minDCF_0_1 = np.concatenate((minDCF_0_1, np.array([compute_min_DCF(ScoresConcat, L, 0.1, 1, 1)])),
                                        axis=0)
            minDCF_0_9 = np.concatenate((minDCF_0_9, np.array([compute_min_DCF(ScoresConcat, L, 0.9, 1, 1)])),
                                        axis=0)

        print(minDCF_0_5.round(3))
        print(minDCF_0_1.round(3))
        print(minDCF_0_9.round(3))

        bayes_error_plot(np.linspace(-2, 2, 21), ScoresConcat, L)
        plt.savefig('plots/bayes_error_plot_SVM_poly_gaussianized_data.jpg')
        plot_ROC(ScoresConcat, L)
        plt.savefig('plots/roc_plot_SVM_poly_gaussianized_data.jpg')
        # plt.show()
    
    # # minDCF plots
    # plt.figure()
    # plt.xscale('log')
    # plt.plot(CArr, minDCF_0_5, label='minDCF-0.5', color='r')
    # plt.plot(CArr, minDCF_0_1, label='minDCF-0.1', color='b')
    # plt.plot(CArr, minDCF_0_9, label='minDCF-0.9', color='g')
    #
    # plt.xlabel('C')
    # plt.ylabel("DCF")
    # plt.tight_layout()
    # plt.legend()
    # # plt.savefig('plots/minDCF_SVM_poly_raw_data.jpg')
    # plt.savefig('plots/minDCF_SVM_poly_gaussianized_data.jpg')
    # # plt.show()
    # sys.exit(0)
    """
    print("#####################################################################################")
    print("SVM RBF Kernel")
    """
    classifierType = "SVM RBF Kernel "
    for C in CRange:
        print("     ########")
        print("     for C = " + str(C))
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                alphaS = train_SVM_Kernel_RBF(DTrain, LTrain, C, 0.01, 0)
                scores = compute_scores_RBF(alphaS, DTrain, LTrain, DTest, 0.01, 0)

                ScoresConcat = np.concatenate((ScoresConcat, scores), axis=0)
                predictedLabels = scores > 0
                LTestConcat = np.concatenate((LTestConcat, predictedLabels), axis=0)

            minDCF_0_5 = np.concatenate((minDCF_0_5, np.array([compute_min_DCF(ScoresConcat, L, 0.5, 1, 1)])),
                                        axis=0)
            minDCF_0_1 = np.concatenate((minDCF_0_1, np.array([compute_min_DCF(ScoresConcat, L, 0.1, 1, 1)])),
                                        axis=0)
            minDCF_0_9 = np.concatenate((minDCF_0_9, np.array([compute_min_DCF(ScoresConcat, L, 0.9, 1, 1)])),
                                        axis=0)

        print(minDCF_0_5.round(3))
        print(minDCF_0_1.round(3))
        print(minDCF_0_9.round(3))

        bayes_error_plot(np.linspace(-2, 2, 21), ScoresConcat, L)
        plt.savefig('plots/bayes_error_plot_SVM_RBF_gaussianized_data.jpg')
        plot_ROC(ScoresConcat, L)
        plt.savefig('plots/roc_plot_SVM_RBF_gaussianized_data.jpg')
        # plt.show()

    # # minDCF plots
    # plt.figure()
    # plt.xscale('log')
    # plt.plot(CArr, minDCF_0_5, label='minDCF-0.5', color='r')
    # plt.plot(CArr, minDCF_0_1, label='minDCF-0.1', color='b')
    # plt.plot(CArr, minDCF_0_9, label='minDCF-0.9', color='g')
    #
    # plt.xlabel('C')
    # plt.ylabel("DCF")
    # plt.tight_layout()
    # plt.legend()
    # # plt.savefig('plots/minDCF_SVM_RBF_raw_data.jpg')
    # plt.savefig('plots/minDCF_SVM_RBF_gaussianized_data.jpg')
    # # plt.show()
    sys.exit(0)
    """
    print("#####################################################################################")
    print("SVM RBF Kernel - different gammas")
    """
    minDCF_0_5_0001 = np.empty([0])
    minDCF_0_5_001 = np.empty([0])
    minDCF_0_5_01 = np.empty([0])
    minDCF_0_5_1 = np.empty([0])

    classifierType = "SVM RBF Kernel - different gammas "
    for C in CRange:
        print("     ########")
        print("     for C = " + str(C))
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            ScoresConcat_0001 = np.empty([0])
            ScoresConcat_001 = np.empty([0])
            ScoresConcat_01 = np.empty([0])
            ScoresConcat_1 = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                # gamma varies between 0.001, 0.01, 0.1, 1
                # gamma = 0.001
                alphaS_0001 = train_SVM_Kernel_RBF(DTrain, LTrain, C, 0.001, 0)
                scores_0001 = compute_scores_RBF(alphaS_0001, DTrain, LTrain, DTest, 0.001, 0)
                ScoresConcat_0001 = np.concatenate((ScoresConcat_0001, scores_0001), axis=0)
                # gamma = 0.01
                alphaS_001 = train_SVM_Kernel_RBF(DTrain, LTrain, C, 0.01, 0)
                scores_001 = compute_scores_RBF(alphaS_001, DTrain, LTrain, DTest, 0.01, 0)
                ScoresConcat_001 = np.concatenate((ScoresConcat_001, scores_001), axis=0)
                # gamma = 0.1
                alphaS_01 = train_SVM_Kernel_RBF(DTrain, LTrain, C, 0.1, 0)
                scores_01 = compute_scores_RBF(alphaS_01, DTrain, LTrain, DTest, 0.1, 0)
                ScoresConcat_01 = np.concatenate((ScoresConcat_01, scores_01), axis=0)
                # gamma = 1
                alphaS_1 = train_SVM_Kernel_RBF(DTrain, LTrain, C, 1, 0)
                scores_1 = compute_scores_RBF(alphaS_1, DTrain, LTrain, DTest, 1, 0)
                ScoresConcat_1 = np.concatenate((ScoresConcat_1, scores_1), axis=0)

            minDCF_0_5_0001 = np.concatenate(
                (minDCF_0_5_0001, np.array([compute_min_DCF(ScoresConcat_0001, L, 0.5, 1, 1)])),
                axis=0)
            minDCF_0_5_001 = np.concatenate(
                (minDCF_0_5_001, np.array([compute_min_DCF(ScoresConcat_001, L, 0.5, 1, 1)])),
                axis=0)
            minDCF_0_5_01 = np.concatenate((minDCF_0_5_01, np.array([compute_min_DCF(ScoresConcat_01, L, 0.5, 1, 1)])),
                                           axis=0)
            minDCF_0_5_1 = np.concatenate((minDCF_0_5_1, np.array([compute_min_DCF(ScoresConcat_1, L, 0.5, 1, 1)])),
                                          axis=0)

    print(CArr.shape)
    print(minDCF_0_5_0001.shape)
    # minDCF plots
    plt.figure()
    plt.xscale('log')
    plt.plot(CArr, minDCF_0_5_0001, label='γ = 0.001', color='r')
    plt.plot(CArr, minDCF_0_5_001, label='γ = 0.01', color='b')
    plt.plot(CArr, minDCF_0_5_01, label='γ = 0.1', color='g')
    plt.plot(CArr, minDCF_0_5_1, label='γ = 1', color='y')

    plt.xlabel('C')
    plt.ylabel("DCF")
    plt.tight_layout()
    plt.legend()
    # plt.savefig('plots/minDCF_SVM_RBF_raw_data.jpg')
    plt.savefig('plots/minDCF_SVM_RBF_gaussianized_data_var_gamma.jpg')
    plt.show()
    sys.exit(0)
    """
