import scipy.special

from functions.cost_computations_functions import *
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


def conf_matrix(llratio, labels, pr, C_fn, C_fp):
    # Computing c* comparing the llr with a threshold t
    t = - np.log((pr * C_fn) / ((1 - pr) * C_fp))

    C_star = np.zeros([llratio.shape[0], ], dtype=int)

    for i in range(llratio.shape[0]):
        if llratio[i] > t:
            C_star[i] = 1
        else:
            C_star[i] = 0

    # Computing the confusion matrix, comparing labels and c*
    conf_matr = compute_conf_matrix_binary(C_star, labels)
    return conf_matr


def weighted_logS(D, gmm_):
    log_S = np.zeros([1, D.shape[1]])
    for i in range(len(gmm_)):
        log_S = log_S + gmm_[i][0] * logpdf_GAU_ND(D, gmm_[i][1], gmm_[i][2])
    return log_S


def compute_log_scores(log_scores):
    S = np.exp(log_scores)
    llr = np.zeros([S.shape[0]])
    for i in range(log_scores.shape[0]):
        llr[i] = np.log(S[i, 1] / S[i, 0])
    return llr


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


def compute_gmm_matrix(DTrain, LTrain, DTest, log_scores, components, iterations):
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
    return log_scores


def GMM_EM_tied(X, gmm):
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
        Sigma_tied = np.zeros((X.shape[0], X.shape[0]))
        Sigma_array = []
        Z_array = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            Z_array.append(Z)
            Sigma_array.append(Sigma)

        for g in range(G):
            Sigma_tied = Sigma_tied + Z_array[g] * Sigma_array[g]

        Sigma_tied = Sigma_tied / N

        U, s, _ = np.linalg.svd(Sigma_tied)
        psi = 0.01
        s[s < psi] = psi
        Sigma_tied = np.dot(U, mcol(s) * U.T)

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            w = Z / N
            mu = mcol(F / Z)
            gmmNew.append((w, mu, Sigma_tied))
        gmm = gmmNew
    return gmm


def compute_gmm_tied_matrix(DTrain, LTrain, DTest, log_scores, components, iterations):
    mean_0 = mean(0, DTrain, LTrain)
    mean_1 = mean(1, DTrain, LTrain)
    covariance_matrix_0 = covariance(0, DTrain, LTrain)
    covariance_matrix_1 = covariance(1, DTrain, LTrain)
    DTrain_0 = DTrain[:, LTrain == 0]
    DTrain_1 = DTrain[:, LTrain == 1]

    N_0 = DTrain_0.shape[1]
    N_1 = DTrain_1.shape[1]
    tied_covariance = (1 / DTrain.shape[1]) * (covariance_matrix_0 * N_0 + covariance_matrix_1 * N_1)

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

            tied_cov_new = tied_covariance
            U, s, _ = np.linalg.svd(tied_cov_new)
            psi = 0.01
            s[s < psi] = psi
            tied_cov_new = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmm_array0.append((weight, mcol(mean_vec0[:, c]), tied_cov_new))

            for c in range(comp):
                gmm_array1.append((weight, mcol(mean_vec1[:, c]), tied_cov_new))

            gmm0 = GMM_EM_tied(DTrain_0, gmm_array0)
            gmm1 = GMM_EM_tied(DTrain_1, gmm_array1)
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

            gmm0 = GMM_EM_tied(DTrain_0, gmm_array0)
            gmm1 = GMM_EM_tied(DTrain_1, gmm_array1)
    iterations = 0

    weighted_logS0 = weighted_logS(DTest, gmm0)
    weighted_logS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weighted_logS0, weighted_logS1), axis=0)
    logS = logS.T
    log_scores = np.concatenate((log_scores, logS))
    return log_scores


def GMM_EM_diag(X, gmm, covariance_matrix_0):
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
            Sigma_diag = Sigma * np.eye(covariance_matrix_0.shape[1])
            U, s, _ = np.linalg.svd(Sigma_diag)
            psi = 0.01
            s[s < psi] = psi
            Sigma_diag = np.dot(U, mcol(s) * U.T)
            gmmNew.append((w, mu, Sigma_diag))
        gmm = gmmNew
    return gmm


def compute_gmm_diag_matrix(DTrain, LTrain, DTest, log_scores, components, iterations):
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

            gmm0 = GMM_EM_diag(DTrain_0, gmm_array0, covariance_matrix_0)
            gmm1 = GMM_EM_diag(DTrain_1, gmm_array1, covariance_matrix_1)
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

            gmm0 = GMM_EM_diag(DTrain_0, gmm_array0, covariance_matrix_0)
            gmm1 = GMM_EM_diag(DTrain_1, gmm_array1, covariance_matrix_1)
    iterations = 0

    weighted_logS0 = weighted_logS(DTest, gmm0)
    weighted_logS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weighted_logS0, weighted_logS1), axis=0)
    logS = logS.T
    log_scores = np.concatenate((log_scores, logS))
    return log_scores


def GMM_EM_tied_diag(X, gmm, covariance_matrix_0):
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
        Sigma_tied = np.zeros((X.shape[0], X.shape[0]))
        Sigma_array = []
        Z_array = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            Sigma = Sigma * np.eye(covariance_matrix_0.shape[1])
            Z_array.append(Z)
            Sigma_array.append(Sigma)

        for g in range(G):
            Sigma_tied = Sigma_tied + Z_array[g] * Sigma_array[g]

        Sigma_tied = Sigma_tied / N

        U, s, _ = np.linalg.svd(Sigma_tied)
        psi = 0.01
        s[s < psi] = psi
        Sigma_tied = np.dot(U, mcol(s) * U.T)

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            w = Z / N
            mu = mcol(F / Z)
            gmmNew.append((w, mu, Sigma_tied))
        gmm = gmmNew
    return gmm


def compute_gmm_tied_diag_matrix(DTrain, LTrain, DTest, log_scores, components, iterations):
    mean_0 = mean(0, DTrain, LTrain)
    mean_1 = mean(1, DTrain, LTrain)
    covariance_matrix_0 = covariance(0, DTrain, LTrain)
    covariance_matrix_1 = covariance(1, DTrain, LTrain)
    DTrain_0 = DTrain[:, LTrain == 0]
    DTrain_1 = DTrain[:, LTrain == 1]

    N_0 = DTrain_0.shape[1]
    N_1 = DTrain_1.shape[1]
    tied_covariance = (1 / DTrain.shape[1]) * (covariance_matrix_0 * N_0 + covariance_matrix_1 * N_1)

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

            tied_cov_new = tied_covariance
            U, s, _ = np.linalg.svd(tied_cov_new)
            psi = 0.01
            s[s < psi] = psi
            tied_cov_new = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmm_array0.append((weight, mcol(mean_vec0[:, c]), tied_cov_new))

            for c in range(comp):
                gmm_array1.append((weight, mcol(mean_vec1[:, c]), tied_cov_new))

            gmm0 = GMM_EM_tied_diag(DTrain_0, gmm_array0, covariance_matrix_0)
            gmm1 = GMM_EM_tied_diag(DTrain_1, gmm_array1, covariance_matrix_1)
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

            gmm0 = GMM_EM_tied_diag(DTrain_0, gmm_array0, covariance_matrix_0)
            gmm1 = GMM_EM_tied_diag(DTrain_1, gmm_array1, covariance_matrix_1)
    iterations = 0

    weighted_logS0 = weighted_logS(DTest, gmm0)
    weighted_logS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weighted_logS0, weighted_logS1), axis=0)
    logS = logS.T
    log_scores = np.concatenate((log_scores, logS))
    return log_scores
