from .functions import *
import numpy as np


class demand_RV:
    # demand_RV("normal", T, omega_0, N, index)
    def __init__(self, dist, T, omega_0, N, seed):
        self.dist = dist
        self.T = T
        self.omega_0 = omega_0
        self.seed = seed
        self.N = N

    def compute_mles(self):
        if self.dist == "normal" or self.dist == "Normal":
            mu, sig = self.omega_0
            Sig = np.zeros((self.T, self.T))
            for t in range(self.T):
                Sig[t, t] = sig[t]
            np.random.seed(self.seed)
            samples = np.random.multivariate_normal(mean=mu, cov=Sig, size=self.N)
            mu_MLE = [np.mean(samples[:, t]) for t in range(self.T)]

            sig_MLE = [np.mean((samples[:, t] - mu_MLE[t]) ** 2) for t in range(self.T)]

            self.mle = (mu_MLE, sig_MLE)
        elif self.dist == "poisson" or self.dist == "Poisson":
            lam = self.omega_0
            np.random.seed(self.seed)
            samples = np.random.poisson(lam=lam, size=(self.N, self.T))

            self.mle = tuple([np.mean(samples[:, t]) for t in range(self.T)])
