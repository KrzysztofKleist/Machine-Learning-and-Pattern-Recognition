import scipy.special

from functions.cost_computations_functions import *
from functions.pca_functions import *


def mean(classID, DTrain, LTrain):
    DC = DTrain[:, LTrain == classID]
    return mcol(DC.mean(1))


def covariance(classID, DTrain, LTrain):
    m = mean(classID, DTrain, LTrain)
    centered_matrix = DTrain[:, LTrain == classID] - m
    N = centered_matrix.shape[1]
    return np.dot(centered_matrix, centered_matrix.T) / N


def logpdf_GAU_ND(X, mean, covariance_matrix):
    M = X.shape[0];
    P = np.linalg.inv(covariance_matrix)
    const = -0.5 * M * np.log(2 * np.pi)
    const += -0.5 * np.linalg.slogdet(covariance_matrix)[1]

    l_x = []
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


def conf_matrix(llratio, labels, pi, Cfn, Cfp):
    t = - np.log((pi * Cfn) / ((1 - pi) * Cfp))

    C_star = np.zeros([llratio.shape[0], ], dtype=int)

    for i in range(llratio.shape[0]):
        if llratio[i] > t:
            C_star[i] = 1
        else:
            C_star[i] = 0

    conf_matr = compute_conf_matrix_binary(C_star, labels)
    return conf_matr


def weighted_logS(D, gmm):
    log_S = np.zeros([1, D.shape[1]])
    for i in range(len(gmm)):
        log_S = log_S + gmm[i][0] * logpdf_GAU_ND(D, gmm[i][1], gmm[i][2])
    return log_S


def compute_log_scores(logS):
    S = np.exp(logS)
    llr = np.zeros([S.shape[0]])
    for i in range(logS.shape[0]):
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


def compute_gmm_matrix(DTrain, LTrain, DTest, logScores, components, iterations):
    mean0 = mean(0, DTrain, LTrain)
    mean1 = mean(1, DTrain, LTrain)
    covMatrix0 = covariance(0, DTrain, LTrain)
    covMatrix1 = covariance(1, DTrain, LTrain)
    DTrain0 = DTrain[:, LTrain == 0]
    DTrain1 = DTrain[:, LTrain == 1]

    gmmArr0 = []
    gmmArr1 = []
    gmm0 = []
    gmm1 = []

    while iterations < np.log2(components):
        iterations = iterations + 1
        weight = 0
        comp = 2 ** iterations
        if iterations == 1:
            weight = 1.0 / comp

            meanVec0 = np.zeros((mean0.shape[0], comp))

            U0, s0, _ = np.linalg.svd(covMatrix0)
            alpha0 = 1
            d0 = U0[:, 0:1] * s0[0] ** 0.5 * alpha0

            U1, s1, _ = np.linalg.svd(covMatrix1)
            alpha1 = 1
            d1 = U1[:, 0:1] * s1[0] ** 0.5 * alpha1

            meanVec0[:, 0] = (mcol(mean0[:, 0]) + d0).ravel()
            meanVec0[:, 1] = (mcol(mean0[:, 0]) - d0).ravel()

            meanVec1 = np.zeros((mean1.shape[0], comp))
            meanVec1[:, 0] = (mcol(mean1[:, 0]) + d1).ravel()
            meanVec1[:, 1] = (mcol(mean1[:, 0]) - d1).ravel()

            covMatrix0New = covMatrix0
            U, s, _ = np.linalg.svd(covMatrix0New)
            psi = 0.01
            s[s < psi] = psi
            covMatrix0New = np.dot(U, mcol(s) * U.T)

            covMatrix1New = covMatrix1
            U, s, _ = np.linalg.svd(covMatrix1New)
            psi = 0.01
            s[s < psi] = psi
            covMatrix1New = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmmArr0.append((weight, mcol(meanVec0[:, c]), covMatrix0New))

            for c in range(comp):
                gmmArr1.append((weight, mcol(meanVec1[:, c]), covMatrix1New))

            gmm0 = GMM_EM(DTrain0, gmmArr0)
            gmm1 = GMM_EM(DTrain1, gmmArr1)
        else:
            gmmArr0 = []
            gmmArr1 = []
            weight0Arr = np.zeros((comp))
            weight1Arr = np.zeros((comp))
            d0Arr = np.zeros((mean0.shape[0], (int(comp / 2))))
            d1Arr = np.zeros((mean1.shape[0], (int(comp / 2))))
            covArr0 = []
            covArr1 = []

            for c in range(int(comp / 2)):
                U0, s0, _ = np.linalg.svd(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                alpha0 = 1
                d0Arr[:, c] = (U0[:, 0:1] * s0[0] ** 0.5 * alpha0).ravel()

            for c in range(int(comp / 2)):
                U1, s1, _ = np.linalg.svd(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                alpha1 = 1
                d1Arr[:, c] = (U1[:, 0:1] * s1[0] ** 0.5 * alpha1).ravel()

            for c in range(int(comp / 2)):
                weight0Arr[2 * c] = gmm0[c][0] / 2
                weight0Arr[(2 * c) + 1] = gmm0[c][0] / 2

            for c in range(int(comp / 2)):
                weight1Arr[2 * c] = gmm1[c][0] / 2
                weight1Arr[(2 * c) + 1] = gmm1[c][0] / 2

            meanVec0 = np.zeros((mean0.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec0[:, 2 * c] = (mcol(gmm0[c][1]) + mcol(d0Arr[:, c])).ravel()
                meanVec0[:, ((2 * c) + 1)] = (mcol(gmm0[c][1]) - mcol(d0Arr[:, c])).ravel()

            meanVec1 = np.zeros((mean1.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec1[:, 2 * c] = (mcol(gmm1[c][1]) + mcol(d1Arr[:, c])).ravel()
                meanVec1[:, ((2 * c) + 1)] = (mcol(gmm1[c][1]) - mcol(d1Arr[:, c])).ravel()

            for c in range(comp):
                gmmArr0.append((weight0Arr[c], mcol(meanVec0[:, c]), covArr0[c]))
            for c in range(comp):
                gmmArr1.append((weight1Arr[c], mcol(meanVec1[:, c]), covArr1[c]))

            gmm0 = GMM_EM(DTrain0, gmmArr0)
            gmm1 = GMM_EM(DTrain1, gmmArr1)
    iterations = 0

    weightedLogS0 = weighted_logS(DTest, gmm0)
    weightedLogS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weightedLogS0, weightedLogS1), axis=0)
    logS = logS.T
    logScores = np.concatenate((logScores, logS))
    return logScores


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
        sigmaTied = np.zeros((X.shape[0], X.shape[0]))
        sigmaArr = []
        Z_array = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            Z_array.append(Z)
            sigmaArr.append(Sigma)

        for g in range(G):
            sigmaTied = sigmaTied + Z_array[g] * sigmaArr[g]

        sigmaTied = sigmaTied / N

        U, s, _ = np.linalg.svd(sigmaTied)
        psi = 0.01
        s[s < psi] = psi
        sigmaTied = np.dot(U, mcol(s) * U.T)

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            w = Z / N
            mu = mcol(F / Z)
            gmmNew.append((w, mu, sigmaTied))
        gmm = gmmNew
    return gmm


def compute_gmm_tied_matrix(DTrain, LTrain, DTest, logScores, components, iterations):
    mean0 = mean(0, DTrain, LTrain)
    mean1 = mean(1, DTrain, LTrain)
    covMatrix0 = covariance(0, DTrain, LTrain)
    covMatrix1 = covariance(1, DTrain, LTrain)
    DTrain0 = DTrain[:, LTrain == 0]
    DTrain1 = DTrain[:, LTrain == 1]

    N_0 = DTrain0.shape[1]
    N_1 = DTrain1.shape[1]
    tied_covariance = (1 / DTrain.shape[1]) * (covMatrix0 * N_0 + covMatrix1 * N_1)

    gmmArr0 = []
    gmmArr1 = []
    gmm0 = []
    gmm1 = []

    while iterations < np.log2(components):
        iterations = iterations + 1
        weight = 0
        comp = 2 ** iterations
        if iterations == 1:
            weight = 1.0 / comp

            meanVec0 = np.zeros((mean0.shape[0], comp))

            U0, s0, _ = np.linalg.svd(covMatrix0)
            alpha0 = 1
            d0 = U0[:, 0:1] * s0[0] ** 0.5 * alpha0

            U1, s1, _ = np.linalg.svd(covMatrix1)
            alpha1 = 1
            d1 = U1[:, 0:1] * s1[0] ** 0.5 * alpha1

            meanVec0[:, 0] = (mcol(mean0[:, 0]) + d0).ravel()
            meanVec0[:, 1] = (mcol(mean0[:, 0]) - d0).ravel()
            meanVec1 = np.zeros((mean1.shape[0], comp))
            meanVec1[:, 0] = (mcol(mean1[:, 0]) + d1).ravel()
            meanVec1[:, 1] = (mcol(mean1[:, 0]) - d1).ravel()

            tied_cov_new = tied_covariance
            U, s, _ = np.linalg.svd(tied_cov_new)
            psi = 0.01
            s[s < psi] = psi
            tied_cov_new = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmmArr0.append((weight, mcol(meanVec0[:, c]), tied_cov_new))

            for c in range(comp):
                gmmArr1.append((weight, mcol(meanVec1[:, c]), tied_cov_new))

            gmm0 = GMM_EM_tied(DTrain0, gmmArr0)
            gmm1 = GMM_EM_tied(DTrain1, gmmArr1)
        else:
            gmmArr0 = []
            gmmArr1 = []
            weight0Arr = np.zeros((comp))
            weight1Arr = np.zeros((comp))
            d0Arr = np.zeros((mean0.shape[0], (int(comp / 2))))
            d1Arr = np.zeros((mean1.shape[0], (int(comp / 2))))
            covArr0 = []
            covArr1 = []

            for c in range(int(comp / 2)):
                U0, s0, _ = np.linalg.svd(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                alpha0 = 1
                d0Arr[:, c] = (U0[:, 0:1] * s0[0] ** 0.5 * alpha0).ravel()

            for c in range(int(comp / 2)):
                U1, s1, _ = np.linalg.svd(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                alpha1 = 1
                d1Arr[:, c] = (U1[:, 0:1] * s1[0] ** 0.5 * alpha1).ravel()

            for c in range(int(comp / 2)):
                weight0Arr[2 * c] = gmm0[c][0] / 2
                weight0Arr[(2 * c) + 1] = gmm0[c][0] / 2

            for c in range(int(comp / 2)):
                weight1Arr[2 * c] = gmm1[c][0] / 2
                weight1Arr[(2 * c) + 1] = gmm1[c][0] / 2

            meanVec0 = np.zeros((mean0.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec0[:, 2 * c] = (mcol(gmm0[c][1]) + mcol(d0Arr[:, c])).ravel()
                meanVec0[:, ((2 * c) + 1)] = (mcol(gmm0[c][1]) - mcol(d0Arr[:, c])).ravel()

            meanVec1 = np.zeros((mean1.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec1[:, 2 * c] = (mcol(gmm1[c][1]) + mcol(d1Arr[:, c])).ravel()
                meanVec1[:, ((2 * c) + 1)] = (mcol(gmm1[c][1]) - mcol(d1Arr[:, c])).ravel()

            for c in range(comp):
                gmmArr0.append((weight0Arr[c], mcol(meanVec0[:, c]), covArr0[c]))
            for c in range(comp):
                gmmArr1.append((weight1Arr[c], mcol(meanVec1[:, c]), covArr1[c]))

            gmm0 = GMM_EM_tied(DTrain0, gmmArr0)
            gmm1 = GMM_EM_tied(DTrain1, gmmArr1)
    iterations = 0

    weightedLogS0 = weighted_logS(DTest, gmm0)
    weightedLogS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weightedLogS0, weightedLogS1), axis=0)
    logS = logS.T
    logScores = np.concatenate((logScores, logS))
    return logScores


def GMM_EM_diag(X, gmm, covMatrix0):
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
            sigmaDiag = Sigma * np.eye(covMatrix0.shape[1])
            U, s, _ = np.linalg.svd(sigmaDiag)
            psi = 0.01
            s[s < psi] = psi
            sigmaDiag = np.dot(U, mcol(s) * U.T)
            gmmNew.append((w, mu, sigmaDiag))
        gmm = gmmNew
    return gmm


def compute_gmm_diag_matrix(DTrain, LTrain, DTest, logScores, components, iterations):
    mean0 = mean(0, DTrain, LTrain)
    mean1 = mean(1, DTrain, LTrain)
    covMatrix0 = covariance(0, DTrain, LTrain)
    covMatrix1 = covariance(1, DTrain, LTrain)
    DTrain0 = DTrain[:, LTrain == 0]
    DTrain1 = DTrain[:, LTrain == 1]

    gmmArr0 = []
    gmmArr1 = []
    gmm0 = []
    gmm1 = []

    while iterations < np.log2(components):
        iterations = iterations + 1
        weight = 0
        comp = 2 ** iterations
        if iterations == 1:
            weight = 1.0 / comp

            meanVec0 = np.zeros((mean0.shape[0], comp))

            U0, s0, _ = np.linalg.svd(covMatrix0)
            alpha0 = 1
            d0 = U0[:, 0:1] * s0[0] ** 0.5 * alpha0

            U1, s1, _ = np.linalg.svd(covMatrix1)
            alpha1 = 1
            d1 = U1[:, 0:1] * s1[0] ** 0.5 * alpha1

            meanVec0[:, 0] = (mcol(mean0[:, 0]) + d0).ravel()
            meanVec0[:, 1] = (mcol(mean0[:, 0]) - d0).ravel()
            meanVec1 = np.zeros((mean1.shape[0], comp))
            meanVec1[:, 0] = (mcol(mean1[:, 0]) + d1).ravel()
            meanVec1[:, 1] = (mcol(mean1[:, 0]) - d1).ravel()

            covMatrix0New = covMatrix0
            U, s, _ = np.linalg.svd(covMatrix0New)
            psi = 0.01
            s[s < psi] = psi
            covMatrix0New = np.dot(U, mcol(s) * U.T)

            covMatrix1New = covMatrix1
            U, s, _ = np.linalg.svd(covMatrix1New)
            psi = 0.01
            s[s < psi] = psi
            covMatrix1New = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmmArr0.append((weight, mcol(meanVec0[:, c]), covMatrix0New))

            for c in range(comp):
                gmmArr1.append((weight, mcol(meanVec1[:, c]), covMatrix1New))

            gmm0 = GMM_EM_diag(DTrain0, gmmArr0, covMatrix0)
            gmm1 = GMM_EM_diag(DTrain1, gmmArr1, covMatrix1)
        else:
            gmmArr0 = []
            gmmArr1 = []
            weight0Arr = np.zeros((comp))
            weight1Arr = np.zeros((comp))
            d0Arr = np.zeros((mean0.shape[0], (int(comp / 2))))
            d1Arr = np.zeros((mean1.shape[0], (int(comp / 2))))
            covArr0 = []
            covArr1 = []

            for c in range(int(comp / 2)):
                U0, s0, _ = np.linalg.svd(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                alpha0 = 1
                d0Arr[:, c] = (U0[:, 0:1] * s0[0] ** 0.5 * alpha0).ravel()

            for c in range(int(comp / 2)):
                U1, s1, _ = np.linalg.svd(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                alpha1 = 1
                d1Arr[:, c] = (U1[:, 0:1] * s1[0] ** 0.5 * alpha1).ravel()

            for c in range(int(comp / 2)):
                weight0Arr[2 * c] = gmm0[c][0] / 2
                weight0Arr[(2 * c) + 1] = gmm0[c][0] / 2

            for c in range(int(comp / 2)):
                weight1Arr[2 * c] = gmm1[c][0] / 2
                weight1Arr[(2 * c) + 1] = gmm1[c][0] / 2

            meanVec0 = np.zeros((mean0.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec0[:, 2 * c] = (mcol(gmm0[c][1]) + mcol(d0Arr[:, c])).ravel()
                meanVec0[:, ((2 * c) + 1)] = (mcol(gmm0[c][1]) - mcol(d0Arr[:, c])).ravel()

            meanVec1 = np.zeros((mean1.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec1[:, 2 * c] = (mcol(gmm1[c][1]) + mcol(d1Arr[:, c])).ravel()
                meanVec1[:, ((2 * c) + 1)] = (mcol(gmm1[c][1]) - mcol(d1Arr[:, c])).ravel()

            for c in range(comp):
                gmmArr0.append((weight0Arr[c], mcol(meanVec0[:, c]), covArr0[c]))
            for c in range(comp):
                gmmArr1.append((weight1Arr[c], mcol(meanVec1[:, c]), covArr1[c]))

            gmm0 = GMM_EM_diag(DTrain0, gmmArr0, covMatrix0)
            gmm1 = GMM_EM_diag(DTrain1, gmmArr1, covMatrix1)
    iterations = 0

    weightedLogS0 = weighted_logS(DTest, gmm0)
    weightedLogS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weightedLogS0, weightedLogS1), axis=0)
    logS = logS.T
    logScores = np.concatenate((logScores, logS))
    return logScores


def GMM_EM_tied_diag(X, gmm, covMatrix0):
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
        sigmaTied = np.zeros((X.shape[0], X.shape[0]))
        sigmaArr = []
        Z_array = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = np.dot(X, (mrow(gamma) * X).T)
            mu = mcol(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)
            Sigma = Sigma * np.eye(covMatrix0.shape[1])
            Z_array.append(Z)
            sigmaArr.append(Sigma)

        for g in range(G):
            sigmaTied = sigmaTied + Z_array[g] * sigmaArr[g]

        sigmaTied = sigmaTied / N

        U, s, _ = np.linalg.svd(sigmaTied)
        psi = 0.01
        s[s < psi] = psi
        sigmaTied = np.dot(U, mcol(s) * U.T)

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            w = Z / N
            mu = mcol(F / Z)
            gmmNew.append((w, mu, sigmaTied))
        gmm = gmmNew
    return gmm


def compute_gmm_tied_diag_matrix(DTrain, LTrain, DTest, logScores, components, iterations):
    mean0 = mean(0, DTrain, LTrain)
    mean1 = mean(1, DTrain, LTrain)
    covMatrix0 = covariance(0, DTrain, LTrain)
    covMatrix1 = covariance(1, DTrain, LTrain)
    DTrain0 = DTrain[:, LTrain == 0]
    DTrain1 = DTrain[:, LTrain == 1]

    N_0 = DTrain0.shape[1]
    N_1 = DTrain1.shape[1]
    tied_covariance = (1 / DTrain.shape[1]) * (covMatrix0 * N_0 + covMatrix1 * N_1)

    gmmArr0 = []
    gmmArr1 = []
    gmm0 = []
    gmm1 = []

    while iterations < np.log2(components):
        iterations = iterations + 1
        weight = 0
        comp = 2 ** iterations
        if iterations == 1:
            weight = 1.0 / comp

            meanVec0 = np.zeros((mean0.shape[0], comp))

            U0, s0, _ = np.linalg.svd(covMatrix0)
            alpha0 = 1
            d0 = U0[:, 0:1] * s0[0] ** 0.5 * alpha0

            U1, s1, _ = np.linalg.svd(covMatrix1)
            alpha1 = 1
            d1 = U1[:, 0:1] * s1[0] ** 0.5 * alpha1

            meanVec0[:, 0] = (mcol(mean0[:, 0]) + d0).ravel()
            meanVec0[:, 1] = (mcol(mean0[:, 0]) - d0).ravel()
            meanVec1 = np.zeros((mean1.shape[0], comp))
            meanVec1[:, 0] = (mcol(mean1[:, 0]) + d1).ravel()
            meanVec1[:, 1] = (mcol(mean1[:, 0]) - d1).ravel()

            tied_cov_new = tied_covariance
            U, s, _ = np.linalg.svd(tied_cov_new)
            psi = 0.01
            s[s < psi] = psi
            tied_cov_new = np.dot(U, mcol(s) * U.T)

            for c in range(comp):
                gmmArr0.append((weight, mcol(meanVec0[:, c]), tied_cov_new))

            for c in range(comp):
                gmmArr1.append((weight, mcol(meanVec1[:, c]), tied_cov_new))

            gmm0 = GMM_EM_tied_diag(DTrain0, gmmArr0, covMatrix0)
            gmm1 = GMM_EM_tied_diag(DTrain1, gmmArr1, covMatrix1)
        else:
            gmmArr0 = []
            gmmArr1 = []
            weight0Arr = np.zeros((comp))
            weight1Arr = np.zeros((comp))
            d0Arr = np.zeros((mean0.shape[0], (int(comp / 2))))
            d1Arr = np.zeros((mean1.shape[0], (int(comp / 2))))
            covArr0 = []
            covArr1 = []

            for c in range(int(comp / 2)):
                U0, s0, _ = np.linalg.svd(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                covArr0.append(gmm0[c][2])
                alpha0 = 1
                d0Arr[:, c] = (U0[:, 0:1] * s0[0] ** 0.5 * alpha0).ravel()

            for c in range(int(comp / 2)):
                U1, s1, _ = np.linalg.svd(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                covArr1.append(gmm1[c][2])
                alpha1 = 1
                d1Arr[:, c] = (U1[:, 0:1] * s1[0] ** 0.5 * alpha1).ravel()

            for c in range(int(comp / 2)):
                weight0Arr[2 * c] = gmm0[c][0] / 2
                weight0Arr[(2 * c) + 1] = gmm0[c][0] / 2

            for c in range(int(comp / 2)):
                weight1Arr[2 * c] = gmm1[c][0] / 2
                weight1Arr[(2 * c) + 1] = gmm1[c][0] / 2

            meanVec0 = np.zeros((mean0.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec0[:, 2 * c] = (mcol(gmm0[c][1]) + mcol(d0Arr[:, c])).ravel()
                meanVec0[:, ((2 * c) + 1)] = (mcol(gmm0[c][1]) - mcol(d0Arr[:, c])).ravel()

            meanVec1 = np.zeros((mean1.shape[0], comp))

            for c in range(int(comp / 2)):
                meanVec1[:, 2 * c] = (mcol(gmm1[c][1]) + mcol(d1Arr[:, c])).ravel()
                meanVec1[:, ((2 * c) + 1)] = (mcol(gmm1[c][1]) - mcol(d1Arr[:, c])).ravel()

            for c in range(comp):
                gmmArr0.append((weight0Arr[c], mcol(meanVec0[:, c]), covArr0[c]))
            for c in range(comp):
                gmmArr1.append((weight1Arr[c], mcol(meanVec1[:, c]), covArr1[c]))

            gmm0 = GMM_EM_tied_diag(DTrain0, gmmArr0, covMatrix0)
            gmm1 = GMM_EM_tied_diag(DTrain1, gmmArr1, covMatrix1)
    iterations = 0

    weightedLogS0 = weighted_logS(DTest, gmm0)
    weightedLogS1 = weighted_logS(DTest, gmm1)
    logS = np.concatenate((weightedLogS0, weightedLogS1), axis=0)
    logS = logS.T
    logScores = np.concatenate((logScores, logS))
    return logScores
