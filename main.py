from functions.gaussian_classifiers_functions import *
from functions.gmm_functions import *
from functions.load_functions import *
from functions.logreg_functions import *
from functions.svm_functions import *

if __name__ == '__main__':
    print("#####################################################################################")
    print("Tied Multivariate Gaussian Classifier, raw data, PCA (m = 9), π~ = 0.5")

    DTrain, LTrain = load('files/Train.txt')
    DTest, LTest = load('files/Test.txt')

    DTrain, P = pca_eval(DTrain, LTrain, 9)
    DTest = np.dot(P.T, DTest)

    _, scores = tiedMulGaussClass(DTrain, DTest, LTrain, LTest)

    CM = compute_conf_matrix_binary(assign_labels(scores, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")

    # plot_ROC(scores, LTest)
    # plt.savefig('plots/EVALUATION_roc_plot_tied_full_cov_raw_data_pca_m_9_pi_0_5.jpg')
    # plt.show()
    # bayes_error_plot(np.linspace(-2, 2, 21), scores, LTest)
    # plt.savefig('plots/EVALUATION_bayes_error_plot_tied_full_cov_raw_data_pca_m_9_pi_0_5.jpg')
    # plt.show()

    print("#####################################################################################")
    print("Logistic Regression, raw data, PCA (m = 9), λ = 1e-4, πT = 0.5, π~ = 0.5 ")

    DTrain, LTrain = load('files/Train.txt')
    DTest, LTest = load('files/Test.txt')

    DTrain, P = pca_eval(DTrain, LTrain, 9)
    DTest = np.dot(P.T, DTest)

    l = 1e-4
    prior_T = 0.5

    x0 = np.zeros(DTrain.shape[0] + 1)
    logRegObj = logRegClass(DTrain, LTrain, l, prior_T)
    v, J, d = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
    w = v[0:-1]
    b = v[-1]
    scores = np.dot(w.T, DTest) + b
    log_reg_scores = scores

    CM = compute_conf_matrix_binary(assign_labels(scores, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")

    # plot_ROC(scores, LTest)
    # plt.savefig('plots/EVALUATION_roc_plot_logReg_raw_data_pca_m_9_lambda_1e-4_piT_0_5_pi_0_5.jpg')
    # plt.show()
    # bayes_error_plot(np.linspace(-2, 2, 21), scores, LTest)
    # plt.savefig('plots/EVALUATION_bayes_error_plot_logReg_raw_data_pca_m_9_lambda_1e-4_piT_0_5_pi_0_5.jpg')
    # plt.show()

    print("#####################################################################################")
    print("SVM Polynomial Kernel, gaussianized data, C = 0.1, d = 2.0, c = 1.0, K = 1.0, no PCA, π~ = 0.5")

    DTrain, LTrain = load('files/Train.txt')
    DTest, LTest = load('files/Test.txt')

    DTrain = gaussianization(DTrain)  # gaussianizes the data, same shape as for preprocessed data
    DTest = gaussianization(DTest)

    alphaS = train_SVM_Kernel_Poly(DTrain, LTrain, 0.1, 2, 1, 1)
    scores = compute_scores_Poly(alphaS, DTrain, LTrain, DTest, 2, 1, 1)

    CM = compute_conf_matrix_binary(assign_labels(scores, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")

    # plot_ROC(scores, LTest)
    # plt.savefig('plots/EVALUATION_roc_plot_SVM_poly_gaussianized_data_pi_0_5.jpg')
    # plt.show()
    # bayes_error_plot(np.linspace(-2, 2, 21), scores, LTest)
    # plt.savefig('plots/EVALUATION_bayes_error_plot_SVM_poly_gaussianized_data_pi_0_5.jpg')
    # plt.show()

    print("#####################################################################################")
    print("GMM tied full covariance, raw data, PCA (m = 9), components = 2")

    DTrain, LTrain = load('files/Train.txt')
    DTest, LTest = load('files/Test.txt')

    DTrain, P = pca_eval(DTrain, LTrain, 9)
    DTest = np.dot(P.T, DTest)

    prior_T = 0.5

    iterations = 0
    components = 2
    log_scores = np.zeros([1, 2])
    log_scores = compute_gmm_tied_matrix(DTrain, LTrain, DTest, log_scores, components, iterations)

    log_scores = log_scores[1:, :]
    # Prior_0 = np.log(1 - prior_T)
    # Prior_1 = np.log(prior_T)
    # logSJoint_0 = log_scores[:, 0] + Prior_0
    # logSJoint_1 = log_scores[:, 1] + Prior_1
    # logSJoint = np.vstack((logSJoint_0, logSJoint_1)).T
    #
    # # scipy.special.logsumexp - log-sum-exp trick
    # logSMarginal = np.reshape(scipy.special.logsumexp(logSJoint, axis=0), (1, logSJoint.shape[1]))
    # logSPost = logSJoint - logSMarginal
    # SPost = np.exp(logSPost)
    # Predicted_labels = np.argmax(SPost, axis=1)

    llr = log_scores[:, 1] - log_scores[:, 0]

    # Compute the confusion matrix and error rates
    # CM = compute_conf_matrix_binary(Predicted_labels, LTest)
    # print(CM)
    # print("minDCF:", compute_min_DCF(llr, LTest, 0.5, 1, 1).round(3))
    # print("actDCF:", compute_act_DCF(llr, LTest, 0.5, 1, 1).round(3))
    # print("error rate:", compute_error_rate(CM).round(4) * 100, " %")
    CM = compute_conf_matrix_binary(assign_labels(llr, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(llr, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(llr, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")

    # plot_ROC(llr, LTest)
    # plt.savefig('plots/EVALUATION_roc_plot_GMM_tied_full_cov_raw_data_pca_m_9_components_2.jpg')
    # plt.show()
    # bayes_error_plot(np.linspace(-2, 2, 21), llr, LTest)
    # plt.savefig('plots/EVALUATION_bayes_error_plot_GMM_tied_full_cov_raw_data_pca_m_9_components_2.jpg')
    # plt.show()

    print("#####################################################################################")
    print("Fusion")

    fusion_scores = llr * 0.5 + scores * 0.5

    CM = compute_conf_matrix_binary(assign_labels(fusion_scores, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(fusion_scores, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(fusion_scores, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")

    plot_ROC(fusion_scores, LTest)
    plt.savefig('plots/EVALUATION_roc_plot_fusion.jpg')
    plt.show()
    bayes_error_plot(np.linspace(-2, 2, 21), fusion_scores, LTest)
    plt.savefig('plots/EVALUATION_bayes_error_plot_fusion.jpg')
    plt.show()
