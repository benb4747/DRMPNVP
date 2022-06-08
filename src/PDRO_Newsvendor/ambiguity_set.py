from functions import *
from demand_RV import *
from scipy.stats import norm, chi2
import itertools as it
import time


class ambiguity_set:
    def __init__(self, demand, set_type, alpha, n_pts, timeout):
        self.demand = demand
        self.set_type = set_type
        self.alpha = alpha
        self.n_pts = n_pts
        self.timeout = timeout
        self.time_taken = 0

    def construct_base_set(self):
        if self.demand.dist in ["normal", "Normal"]:
            start = time.perf_counter()
            omega_0 = self.demand.omega_0
            z = norm.ppf(1 - self.alpha / 2)
            N = self.demand.N
            omega_hat = self.demand.mle
            mu_hat, sig_hat = omega_hat
            mu_CIs = [
                (
                    mu_hat[t] - z * (sig_hat[t] / np.sqrt(N)),
                    mu_hat[t] + z * (sig_hat[t] / np.sqrt(N)),
                )
                for t in range(self.demand.T)
            ]
            sig_CIs = [
                (
                    sig_hat[t] - z * (sig_hat[t] / np.sqrt(2 * N)),
                    sig_hat[t] + z * (sig_hat[t] / np.sqrt(2 * N)),
                )
                for t in range(self.demand.T)
            ]

            mu_CIs_disc = [
                [
                    mu_CIs[t][0] + (j / (self.n_pts - 1)) * (mu_CIs[t][1] - mu_CIs[t][0])
                    for j in range(self.n_pts)
                ]
                for t in range(self.demand.T)
            ]

            sig_CIs_disc = [
                [
                    sig_CIs[t][0] + (j / (self.n_pts - 1)) * (sig_CIs[t][1] - sig_CIs[t][0])
                    for j in range(self.n_pts)
                ]
                for t in range(self.demand.T)
            ]

            mu_vals = list(it.product(*mu_CIs_disc))
            sig_vals = list(it.product(*sig_CIs_disc))

            Omega = [(mu, sig) for mu in mu_vals for sig in sig_vals]
            end = time.perf_counter()

        elif self.demand.dist in ["Poisson", "poisson"]:
            start = time.perf_counter()
            lam_0 = self.demand.omega_0
            z = norm.ppf(1 - self.alpha/2)
            N = self.demand.N
            lam_hat = self.demand.mle
            CIs = [(lam_hat[t] - z * (lam_hat[t] / np.sqrt(N)), lam_hat[t] + z * (lam_hat[t] / np.sqrt(N)))
                      for t in range(self.demand.T)]

            CIs_disc = [[CIs[t][0] + (j / (self.n_pts - 1)) * (CIs[t][1] - CIs[t][0]) 
                         for j in range(self.n_pts)]
                        for t in range(self.demand.T)]

            Omega = list(it.product(*CIs_disc))
            end = time.perf_counter()

        self.time_taken = end - start
        if self.time_taken > self.timeout:
            self.base_set = "T.O."
        else:
            self.base_set = Omega

    def construct_confidence_set(self):
        left = self.timeout - self.time_taken
        start = time.perf_counter()
        if self.demand.dist in ["Normal", "normal"]:
            if self.demand.dist == "normal":
                chi = chi2.ppf(q=1 - self.alpha, df=2 * self.demand.T)
                confset = []
                for omega in self.base_set:
                    if (
                        norm_conf_val(self.demand.N, self.demand.T, omega, self.demand.mle)
                        <= chi
                    ):
                        confset.append(omega)
                    tt = time.perf_counter() - start
                    if tt > left:
                        self.confidence_set_full = "T.O."
                        return

                self.confidence_set_full = confset
                self.time_taken += time.perf_counter() - start

        if self.demand.dist in ["Poisson", "poisson"]:
            chi = chi2.ppf(q=1 - self.alpha, df=2 * self.demand.T)
            confset = []
            for lam in self.base_set:
                if (
                    pois_conf_val(self.demand.N, self.demand.T, lam, self.demand.mle)
                    <= chi
                ):
                    confset.append(omega)
                tt = time.perf_counter() - start
                if tt > left:
                    self.confidence_set_full = "T.O."
                    return

            self.confidence_set_full = confset
            self.time_taken += time.perf_counter() - start

        if tuple(self.demand.mle) not in self.confidence_set_full:
            self.confidence_set_full.append(tuple(self.demand.mle))

    def reduce(self):
        if self.demand.dist in ["Normal", "normal"]:
            left = self.timeout - self.time_taken
            start = time.perf_counter()
            AS_reduced = self.confidence_set_full.copy()
            for omega in AS_reduced:
                if time.perf_counter() - start > left:
                    self.reduced = "T.O."
                    self.time_taken += time.perf_counter() - start
                    return
                mu, sig = omega
                same_mu = [o for o in AS_reduced if o[0] == mu]
                # similar_sig = [o for o in same_mu if np.where(np.array(sig) != np.array(o[1]))[0].shape[0] == 1]
                for o in same_mu:
                    diff = np.where(np.array(sig) != np.array(o[1]))[0]
                    if diff.shape[0] == 1:
                        diff_ind = diff[0]
                        # print(diff_ind)
                        if sig[diff_ind] >= o[1][diff_ind]:  # o is dominated by omega
                            AS_reduced.remove(o)
            reduced = []
            for o in self.confidence_set_full:
                if o in AS_reduced:
                    reduced.append(o)
            self.reduced = reduced
            end = time.perf_counter()
            self.time_taken += end - start

        elif self.demand.dist in ["Poisson", "poisson"]:
            # not sure how this will work yet
            reduced = self.confidence_set_full 
        self.reduced = reduced

    def compute_extreme_distributions(self):
        start = time.perf_counter()
        left = self.timeout - self.time_taken
        if self.reduced == None:
            self.reduce()
        if self.demand.dist in ["normal", "Normal"]:
            M = []
            T = len(self.confidence_set_full[0][0])
            for o in self.confidence_set_full:
                if o[0] not in M:
                    M.append(o[0])

            ext_1 = []
            for mu in M:
                max_sig_sum = max(
                    [sum(o[1]) for o in self.confidence_set_full if o[0] == mu]
                )
                ext_1 += [
                    o
                    for o in self.confidence_set_full
                    if o[0] == mu and sum(o[1]) == max_sig_sum
                ]

            S = []
            for o in ext_1:
                if o[1] not in S:
                    S.append(o[1])

            ext_2 = []
            for sig in S:
                mu_max = [
                    max([mu[t] for (mu, s) in ext_1 if s == sig])
                    for t in range(self.demand.T)
                ]
                mu_min = [
                    min([mu[t] for (mu, s) in ext_1 if s == sig])
                    for t in range(self.demand.T)
                ]

                ext_2 += [
                    (m, s)
                    for (m, s) in ext_1
                    if s == sig
                    and np.any(
                        np.array(
                            [
                                m[t] in [mu_max[t], mu_min[t]]
                                for t in range(self.demand.T)
                            ]
                        )
                    )
                ]
            self.extreme_distributions = ext_2
            end = time.perf_counter()
            self.time_taken += end - start
            if self.time_taken > self.timeout:
                self.extreme_distributions = "T.O."

            return

        elif self.demand.dist in ["Poisson", "poisson"]:
            start = time.perf_counter()
            lam_max = [max(lam[t] for lam in self.reduced) for t in range(self.demand.T)]
            lam_min = [min(lam[t] for lam in self.reduced) for t in range(self.demand.T)]
            
            self.extreme_distributions = [lam for lam in self.reduced if
                                         np.any(np.array([lam[t] in [lam_max[t], lam_min[t]] 
                                                          for t in range(self.demand.T)]))
                                         ]
            end = time.perf_counter()
            self.time_taken += end - start
            if self.time_taken > self.timeout:
                self.extreme_distributions = "T.O."

            return