import scipy.special

from functions.cost_computations_functions import *
from functions.kfolds_functions import *
from functions.load_functions import *
from functions.pca_functions import *


def mean(class_identifier, training_data, training_labels):
    Dc = training_data[:, training_labels == class_identifier]
    return mcol(Dc.mean(1))


def covariance(class_identifier, training_data, training_labels):
    m = mean(class_identifier, training_data, training_labels)
    centered_matrix = training_data[:, training_labels == class_identifier] - m
    N = centered_matrix.shape[1]
    return np.dot(centered_matrix, centered_matrix.T) / N


def logpdf_GAU_ND(X, mean, covariance_matrix):
    M = X.shape[0];
    P = np.linalg.inv(covariance_matrix)
    const = -0.5 * M * np.log(2 * np.pi)
    const += -0.5 * np.linalg.slogdet(covariance_matrix)[1]

    l_x = [];
    for i in range(X.shape[1]):
        x = X[:, i:i + 1]
        res = const - 0.5 * np.dot((x - mean).T, np.dot(P, (x - mean)))
        l_x.append(res)

    return mrow(np.array(l_x))


def GMM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)


def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        P = np.exp(SJ - SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            psi = 0.01
            s[s < psi] = psi
            Sigma = np.dot(U, mcol(s) * U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
    return gmm


def conf_matrix(llratio, labs, pr, C_fn, C_fp):
    # Computing c* comparing the llr with a threshold t
    t = - np.log((pr * C_fn) / ((1 - pr) * C_fp))

    C_star = np.zeros([llratio.shape[0], ], dtype=int)

    for i in range(llratio.shape[0]):
        if llratio[i] > t:
            C_star[i] = 1
        else:
            C_star[i] = 0

    # Computing the confusion matrix, comparing labels and c*
    conf_matr = np.zeros([2, 2], dtype=int)
    for j in range(2):
        for i in range(2):
            conf_matr[j, i] = ((C_star == j) * (labs == i)).sum()
    return conf_matr


def weighted_logS(D, gmm_):
    log_S = np.zeros([1, D.shape[1]])
    for i in range(len(gmm_)):
        log_S = log_S + gmm_[i][0] * logpdf_GAU_ND(D, gmm_[i][1], gmm_[i][2])
    return log_S


if __name__ == '__main__':
    D, L = load('files/Train.txt')

    m = 9
    # D = pca(D, L, m)

    kRange = [5]
    # pcaMRange = [8]
    pcaMRange = [9]
    # pcaMRange = [8, 9]

    prior_T = 0.5  # change between 0.5, 0.1, 0.9

    print("#####################################################################################")
    print("GMM")

    components = 8
    iterations = 0

    real_labels = []
    log_scores = np.zeros([1, 2])

    classifierType = "GMM "
    for k in kRange:
        print("k: {}".format(k), "\r", end="")
        for iter in range(k):
            DTrain, DTest, LTrain, LTest = kFolds(D, L, k, iter)

            mean_0 = mean(0, DTrain, LTrain)
            mean_1 = mean(1, DTrain, LTrain)
            covariance_matrix_0 = covariance(0, DTrain, LTrain)
            covariance_matrix_1 = covariance(1, DTrain, LTrain)
            DTrain_0 = DTrain[:, LTrain == 0]
            DTrain_1 = DTrain[:, LTrain == 1]

            gmm_array0 = []
            gmm_array1 = []
            gmm0 = []
            gmm1 = []

            while iterations < np.log2(components):
                iterations = iterations + 1
                weight = 0
                comp = 2 ** iterations
                if iterations == 1:
                    weight = 1.0 / comp

                    mean_vec0 = np.zeros((mean_0.shape[0], comp))

                    U0, s0, _ = np.linalg.svd(covariance_matrix_0)
                    alpha0 = 1
                    d0 = U0[:, 0:1] * s0[0] ** 0.5 * alpha0

                    U1, s1, _ = np.linalg.svd(covariance_matrix_1)
                    alpha1 = 1
                    d1 = U1[:, 0:1] * s1[0] ** 0.5 * alpha1

                    mean_vec0[:, 0] = (mcol(mean_0[:, 0]) + d0).ravel()
                    mean_vec0[:, 1] = (mcol(mean_0[:, 0]) - d0).ravel()
                    mean_vec1 = np.zeros((mean_1.shape[0], comp))
                    mean_vec1[:, 0] = (mcol(mean_1[:, 0]) + d1).ravel()
                    mean_vec1[:, 1] = (mcol(mean_1[:, 0]) - d1).ravel()

                    cov_matr0_New = covariance_matrix_0
                    U, s, _ = np.linalg.svd(cov_matr0_New)
                    psi = 0.01
                    s[s < psi] = psi
                    cov_matr0_New = np.dot(U, mcol(s) * U.T)

                    cov_matr1_New = covariance_matrix_1
                    U, s, _ = np.linalg.svd(cov_matr1_New)
                    psi = 0.01
                    s[s < psi] = psi
                    cov_matr1_New = np.dot(U, mcol(s) * U.T)

                    for c in range(comp):
                        gmm_array0.append((weight, mcol(mean_vec0[:, c]), cov_matr0_New))

                    for c in range(comp):
                        gmm_array1.append((weight, mcol(mean_vec1[:, c]), cov_matr1_New))

                    gmm0 = GMM_EM(DTrain_0, gmm_array0)
                    gmm1 = GMM_EM(DTrain_1, gmm_array1)
                else:
                    gmm_array0 = []
                    gmm_array1 = []
                    weight0_array = np.zeros((comp))
                    weight1_array = np.zeros((comp))
                    d0_array = np.zeros((mean_0.shape[0], (int(comp / 2))))
                    d1_array = np.zeros((mean_1.shape[0], (int(comp / 2))))
                    cov_array0 = []
                    cov_array1 = []

                    for c in range(int(comp / 2)):
                        U0, s0, _ = np.linalg.svd(gmm0[c][2])
                        cov_array0.append(gmm0[c][2])
                        cov_array0.append(gmm0[c][2])
                        alpha0 = 1
                        d0_array[:, c] = (U0[:, 0:1] * s0[0] ** 0.5 * alpha0).ravel()

                    for c in range(int(comp / 2)):
                        U1, s1, _ = np.linalg.svd(gmm1[c][2])
                        cov_array1.append(gmm1[c][2])
                        cov_array1.append(gmm1[c][2])
                        alpha1 = 1
                        d1_array[:, c] = (U1[:, 0:1] * s1[0] ** 0.5 * alpha1).ravel()

                    for c in range(int(comp / 2)):
                        weight0_array[2 * c] = gmm0[c][0] / 2
                        weight0_array[(2 * c) + 1] = gmm0[c][0] / 2

                    for c in range(int(comp / 2)):
                        weight1_array[2 * c] = gmm1[c][0] / 2
                        weight1_array[(2 * c) + 1] = gmm1[c][0] / 2

                    mean_vec0 = np.zeros((mean_0.shape[0], comp))

                    for c in range(int(comp / 2)):
                        mean_vec0[:, 2 * c] = (mcol(gmm0[c][1]) + mcol(d0_array[:, c])).ravel()
                        mean_vec0[:, ((2 * c) + 1)] = (mcol(gmm0[c][1]) - mcol(d0_array[:, c])).ravel()

                    mean_vec1 = np.zeros((mean_1.shape[0], comp))

                    for c in range(int(comp / 2)):
                        mean_vec1[:, 2 * c] = (mcol(gmm1[c][1]) + mcol(d1_array[:, c])).ravel()
                        mean_vec1[:, ((2 * c) + 1)] = (mcol(gmm1[c][1]) - mcol(d1_array[:, c])).ravel()

                    for c in range(comp):
                        gmm_array0.append((weight0_array[c], mcol(mean_vec0[:, c]), cov_array0[c]))
                    for c in range(comp):
                        gmm_array1.append((weight1_array[c], mcol(mean_vec1[:, c]), cov_array1[c]))

                    gmm0 = GMM_EM(DTrain_0, gmm_array0)
                    gmm1 = GMM_EM(DTrain_1, gmm_array1)
            iterations = 0

            weighted_logS0 = weighted_logS(DTest, gmm0)
            weighted_logS1 = weighted_logS(DTest, gmm1)
            logS = np.concatenate((weighted_logS0, weighted_logS1), axis=0)
            logS = logS.T
            log_scores = np.concatenate((log_scores, logS))
            real_labels = np.concatenate((real_labels, LTest), axis=0)

        log_scores = log_scores[1:, :]
        Pc_0 = np.log(1 - prior_T)
        Pc_1 = np.log(prior_T)
        logSJoint_0 = log_scores[:, 0] + Pc_0
        logSJoint_1 = log_scores[:, 1] + Pc_1
        logSJoint = np.vstack((logSJoint_0, logSJoint_1)).T

        # The function scipy.special.logsumexp permits us to compute the log-sum-exp trick
        logSMarginal = np.reshape(scipy.special.logsumexp(logSJoint, axis=0), (1, logSJoint.shape[1]))
        logSPost = logSJoint - logSMarginal

        # Do the exponential operation for the Posterior Probabilities since we want to do a comparison with the
        # initial validation set
        SPost = np.exp(logSPost)
        # The function ArgMax returns the indices of the maximum values along the indicated axis
        # In our case we have two columns, one for each class and we want to find the Maximum Likelihood for each
        # sample (each row of the matrix)
        Predicted_labels = np.argmax(SPost, axis=1)

        # Compute the confusion matrix, accurancy and error rates
        CM = compute_conf_matrix_binary(Predicted_labels, real_labels)
        print("Model error:", compute_error_rate(CM).round(3) * 100, "%")
        print("Confusion matrix: ")
        print(CM)

        # Compute the log-likelihood ratio llr by using the score matrix
        S = np.exp(log_scores)
        llr = np.zeros([S.shape[0]])
        for i in range(log_scores.shape[0]):
            llr[i] = np.log(S[i, 1] / S[i, 0])
        thresholds = np.array(llr)
        scores = S[:, 1] / S[:, 0]

        # Compute the normalized DCF of our model with a threshold that is our prior_T for computing the rates
        Cfn = 1
        Cfp = 1
        # Compute the confusion matrix with the llr calculated previously and with real_labels from the k fold
        confusion_matrix = conf_matrix(llr, real_labels, prior_T, Cfn, Cfp)

        print("Actual DCF", compute_act_DCF(scores, real_labels, 0.5, 1.0, 1.0))
        print("Actual normalized DCF", compute_normalized_emp_Bayes(confusion_matrix, 0.5, 1.0, 1.0))
        print("Minimum normalized DCF:", compute_min_DCF(scores, real_labels, 0.5, 1.0, 1.0))

        plot_ROC(scores, real_labels)
        plt.show()

        # Bayes error plot
        bayes_error_plot(np.linspace(-1, 1, 21), scores, real_labels)
        plt.show()
