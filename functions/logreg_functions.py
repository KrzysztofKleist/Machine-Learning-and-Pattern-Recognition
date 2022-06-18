import numpy as np


class logRegClass:
    def __init__(self, DTR, LTR, l, prior_t=0.5):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.prior_t = prior_t
        self.Z = LTR * 2.0 - 1.0
        self.M = DTR.shape[0]

    def logreg_obj(self, v):
        w = v[0:self.M]
        b = v[-1]
        S = np.dot(w.T, self.DTR)
        S = S + b
        crossEntropy_good = np.logaddexp(0, -S[self.Z > 0] * self.Z[self.Z > 0]).mean()
        crossEntropy_bad = np.logaddexp(0, -S[self.Z < 0] * self.Z[self.Z < 0]).mean()
        return 0.5 * self.l * np.linalg.norm(w) ** 2 + self.prior_t * crossEntropy_good + (
                1 - self.prior_t) * crossEntropy_bad
