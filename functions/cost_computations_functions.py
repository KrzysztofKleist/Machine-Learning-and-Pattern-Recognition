import numpy as np
from matplotlib import pyplot as plt


def compute_conf_matrix_binary(Pred, Labels):
    C = np.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (Labels == 1)).sum()
    return C


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -np.log(pi * Cfn) + np.log((1 - pi) * Cfp)
    P = scores > th
    return np.int32(P)


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi * Cfn, (1 - pi) * Cfp)


def compute_min_norm_DCF(scores, LTE, prior_tilde, Cfn, Cfp):
    Bayes_emp_risk_dummy = min(prior_tilde * Cfn, (1 - prior_tilde) * Cfp)
    thresholds = np.array(scores)
    Bayes_emp = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(scores > t)
        Conf = np.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                Conf[j, i] = ((Pred == j) * (LTE == i)).sum()
        FNR_minDCF = Conf[0, 1] / (Conf[0, 1] + Conf[1, 1])
        FPR_minDCF = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])
        Bayes_emp[idx] = prior_tilde * Cfn * FNR_minDCF + (1 - prior_tilde) * Cfp * FPR_minDCF

    Bayes_emp_min = Bayes_emp.min()
    min_normDCF = Bayes_emp_min / Bayes_emp_risk_dummy
    return min_normDCF


def compute_error_rate(CM):
    return (CM[0][1] + CM[1][0]) / CM.sum()


def bayes_error_plot_values(pArray, scores, labels, minCost=False):
    y = []
    for p in pArray:
        pi = 1.0 / (1.0 + np.exp(-p))
        # print(pi)
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return np.array(y)


def bayes_error_plot(pArray, scores, labels):
    plt.figure()
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.plot(pArray, bayes_error_plot_values(pArray, scores, labels, minCost=False), color='r', label="actDCF")
    plt.plot(pArray, bayes_error_plot_values(pArray, scores, labels, minCost=True), color='b', label="minDCF")
    plt.legend()


def plot_ROC(scores, labels):
    thresholds = np.array(scores)
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(scores > t)
        Conf = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Conf[i, j] = ((Pred == i) * (labels == j)).sum()
        TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1])
        FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])

    plt.figure()
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(FPR, TPR)
