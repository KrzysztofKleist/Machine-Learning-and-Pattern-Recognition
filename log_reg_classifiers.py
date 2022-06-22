import scipy

from functions.cost_computations_functions import *
from functions.kfolds_functions import *
from functions.load_functions import *
from functions.logreg_functions import *
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
    pcaMRange = [9]
    # pcaMRange = [8, 9]

    # lRange for minDCF plots
    # lRange = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10, 100, 1e3, 1e4, 1e5, 1e6]
    lRange = [1e-4]

    prior_T = 0.5  # change between 0.5, 0.1, 0.9

    lArr = np.array(lRange)
    minDCF_0_5 = np.empty([0])
    minDCF_0_1 = np.empty([0])
    minDCF_0_9 = np.empty([0])

    print("#####################################################################################")
    print("Logistic Regression")

    classifierType = "Logistic Regression "
    for l in lRange:
        print("     ########")
        print("     for l = " + str(l))
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            LTestConcat = np.empty([0])
            ScoresConcat = np.empty([0])
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                x0 = np.zeros(DTrain.shape[0] + 1)
                logRegObj = logRegClass(DTrain, LTrain, l, prior_T)
                v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
                w = v[0:-1]
                b = v[-1]
                STE = np.dot(w.T, DTest) + b
                ScoresConcat = np.concatenate((ScoresConcat, STE), axis=0)
                predictedLabels = STE > 0
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

    # # minDCF plots
    # plt.figure()
    # plt.xscale('log')
    # plt.plot(lArr, minDCF_0_5, label='minDCF-0.5', color='r')
    # plt.plot(lArr, minDCF_0_1, label='minDCF-0.1', color='b')
    # plt.plot(lArr, minDCF_0_9, label='minDCF-0.9', color='g')
    #
    # plt.xlabel('λ')
    # plt.ylabel("DCF")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig('plots/minDCF_logReg_raw_data.jpg')
    # # plt.savefig('plots/minDCF_logReg_gaussianized_data.jpg')
    # plt.show()
    # sys.exit(0)

    ###### THIS CODE IS USELESS ######
    # print("#####################################################################################")
    # print("Logistic Regression + PCA")
    """
    classifierType = "Logistic Regression + PCA "
    for m in pcaMRange:
        print(" #################")
        print(" PCA for m = " + str(m))
        DP = pca(D, L, m)
        minDCF_0_5 = np.empty([0])
        minDCF_0_1 = np.empty([0])
        minDCF_0_9 = np.empty([0])
        for l in lRange:
            print("     ########")
            print("     for l = " + str(l))
            for k in kRange:
                print("k: {}".format(k), "\r", end="")
                LTestConcat = np.empty([0])
                ScoresConcat = np.empty([0])
                for iter in range(k):
                    DTrain, DTest, LTrain, LTest = kFolds(DP, L, k, iter)
                    x0 = np.zeros(DTrain.shape[0] + 1)
                    logRegObj = logRegClass(DTrain, LTrain, l, prior_T)
                    v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
                    w = v[0:-1]
                    b = v[-1]
                    STE = np.dot(w.T, DTest) + b
                    ScoresConcat = np.concatenate((ScoresConcat, STE), axis=0)
                    predictedLabels = STE > 0
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
            plt.savefig('plots/bayes_error_plot_log_reg_raw_data_pca_m_9_pi_0_5.jpg')
            plt.show()
            plot_ROC(ScoresConcat, L)
            plt.savefig('plots/roc_plot_log_reg_raw_data_pca_m_9_pi_0_5.jpg')
            plt.show()

            sys.exit(0)
    """
