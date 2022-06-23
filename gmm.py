from functions.gmm_functions import *
from functions.kfolds_functions import *
from functions.load_functions import *

if __name__ == '__main__':
    D, L = load('files/Train.txt')

    # comment next two lines to get raw features
    Dg = gaussianization(D)  # gaussianizes the data, same shape as for preprocessed data
    D = Dg

    m = 8
    D = pca(D, L, m)

    kRange = [5]

    prior_T = 0.5  # change between 0.5, 0.1, 0.9

    print("#####################################################################################")
    print("GMM")

    # componentsRange = [1, 2, 4, 8, 16, 32]
    componentsRange = [2]

    for components in componentsRange:
        print("##########")
        print("components: ", components)

        iterations = 0
        log_scores = np.zeros([1, 2])

        classifierType = "GMM "
        for k in kRange:
            print("k: {}".format(k), "\r", end="")
            for iter in range(k):
                DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)
                log_scores = compute_gmm_matrix(DTrain, LTrain, DTest, log_scores, components,
                                                iterations)

            log_scores = log_scores[1:, :]
            Prior_0 = np.log(1 - prior_T)
            Prior_1 = np.log(prior_T)
            logSJoint_0 = log_scores[:, 0] + Prior_0
            logSJoint_1 = log_scores[:, 1] + Prior_1
            logSJoint = np.vstack((logSJoint_0, logSJoint_1)).T

            # scipy.special.logsumexp - log-sum-exp trick
            logSMarginal = np.reshape(scipy.special.logsumexp(logSJoint, axis=0), (1, logSJoint.shape[1]))
            logSPost = logSJoint - logSMarginal
            SPost = np.exp(logSPost)
            Predicted_labels = np.argmax(SPost, axis=1)

            # Compute the confusion matrix and error rates
            CM = compute_conf_matrix_binary(Predicted_labels, L)
            # print("Model error:", compute_error_rate(CM).round(3) * 100, "%")
            # print("Confusion matrix: ")
            # print(CM)

            # Compute the score
            llr = log_scores[:, 1] - log_scores[:, 0]

            print("Actual DCF", compute_act_DCF(llr, L, 0.5, 1.0, 1.0))
            # print("Actual normalized DCF", compute_normalized_emp_Bayes(CM, 0.5, 1.0, 1.0))
            print("Minimum normalized DCF:", compute_min_DCF(llr, L, 0.5, 1.0, 1.0).round(3))

            plot_ROC(llr, L)
            # plt.savefig('plots/roc_plot_gmm_full_cov_raw_data_pca_m_9_components_2.jpg')
            plt.savefig('plots/roc_plot_gmm_full_cov_gaussianized_data_pca_m_8_components_2.jpg')
            plt.show()

            # Bayes error plot
            bayes_error_plot(np.linspace(-2, 2, 21), llr, L)
            # plt.savefig('plots/bayes_error_plot_gmm_full_cov_raw_data_pca_m_9_components_2.jpg')
            plt.savefig('plots/bayes_error_plot_gmm_full_cov_gaussianized_data_pca_m_8_components_2.jpg')
            plt.show()
