import numpy as np


def kFolds(D, L, k, n):  # k - number of folds, n - current fold chosen to be evaluation set
    interval = int(np.ceil(D.shape[1] / k))

    DTest = D[:, (n * interval): ((n + 1) * interval)]
    DTrain1 = D[:, : (n * interval)]
    DTrain2 = D[:, ((n + 1) * interval):]
    DTrain = np.concatenate((DTrain1, DTrain2), axis=1)

    LTest = L[(n * interval): ((n + 1) * interval)]
    LTrain1 = L[: (n * interval)]
    LTrain2 = L[((n + 1) * interval):]
    LTrain = np.concatenate((LTrain1, LTrain2), axis=0)

    return DTrain, DTest, LTrain, LTest
