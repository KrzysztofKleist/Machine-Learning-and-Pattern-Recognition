from functions.cost_computations_functions import *
from functions.gaussian_classifiers_functions import *
from functions.load_functions import *
from functions.pca_functions import *

if __name__ == '__main__':
    print("#####################################################################################")
    print("Tied Multivariate Gaussian Classifier, raw data, PCA (m = 9), π tilde = 0.5")

    DTrain, LTrain = load('files/Train.txt')
    DTest, LTest = load('files/Test.txt')

    DTrain, P = pca_eval(DTrain, LTrain, 9)
    DTest = np.dot(P.T, DTest)

    LPred, scores = tiedMulGaussClass(DTrain, DTest, LTrain, LTest)
    CM = compute_conf_matrix_binary(assign_labels(scores, 0.5, 1, 1), LTest)
    print(CM)
    print("minDCF:", compute_min_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("actDCF:", compute_act_DCF(scores, LTest, 0.5, 1, 1).round(3))
    print("error rate:", compute_error_rate(CM).round(4) * 100, " %")
    plot_ROC(scores, LTest)
    plt.savefig('plots/EVALUATION_roc_plot_tied_full_cov_raw_data_pca_m_9_pi_0_5.jpg')
    plt.show()
    bayes_error_plot(np.linspace(-2, 2, 21), scores, LTest)
    plt.savefig('plots/EVALUATION_bayes_error_plot_tied_full_cov_raw_data_pca_m_9_pi_0_5.jpg')
    plt.show()

