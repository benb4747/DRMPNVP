from .functions import *
from .demand_RV import *
from scipy.stats import norm, chi2
import itertools as it
import time
from numba import njit


@njit
def reduce_njit(AS_conf, sigs_t, red):
    y = AS_conf[0]
    n = len(y)
    T = int(len(y) / 2)
    c = 0
    for k in range(AS_conf.shape[0]):
        x = AS_conf[k]
        dom = False
        for t in range(T):
            if x[T + t] != max(sigs_t[t]):
                j = 0
                while sigs_t[t, j] != x[T + t]:
                    j += 1
                j1 = j
                while sigs_t[t, j] != 0:
                    j += 1
                j2 = j
                for i in sigs_t[t, j1 + 1 : j2]:
                    x_ = np.zeros(n)
                    for k in range(2 * T):
                        x_[k] = x[k]
                    x_[T + t] = i
                    for e in range(AS_conf.shape[0]):
                        same = True
                        for f in range(AS_conf.shape[1]):
                            if AS_conf[e, f] != x_[f]:
                                same = False
                                break
                        if same:
                            dom = True
                            break
                    if dom:
                        break

            if dom:
                break

        if not dom:
            red[c] = x
            c += 1
    return red[:c]


class ambiguity_set:
    def __init__(self, demand, set_type, alpha, n_pts, timeout):
        self.demand = demand
        self.set_type = set_type
        self.alpha = alpha
        self.n_pts = n_pts
        self.timeout = timeout
        self.time_taken = 0
        self.reduced = None

    def construct_base_set(self):
        if self.demand.dist in ["normal", "Normal"]:
            start = time.perf_counter()
            omega_0 = self.demand.omega_0
            # z = norm.ppf(1 - self.alpha / 2)
            z = np.sqrt(chi2.ppf(q=1 - self.alpha, df=2 * self.demand.T))
            N = self.demand.N
            omega_hat = self.demand.mle
            mu_hat, sig_hat = omega_hat
            mu_CIs = [
                (
                    max(mu_hat[t] - z * (sig_hat[t] / np.sqrt(N)), 0),
                    mu_hat[t] + z * (sig_hat[t] / np.sqrt(N)),
                )
                for t in range(self.demand.T)
            ]
            sig_CIs = [
                (
                    max(sig_hat[t] - z * (sig_hat[t] / np.sqrt(2 * N)), 0),
                    sig_hat[t] + z * (sig_hat[t] / np.sqrt(2 * N)),
                )
                for t in range(self.demand.T)
            ]

            mu_CIs_disc = [
                [
                    mu_CIs[t][0]
                    + (j / (self.n_pts - 1)) * (mu_CIs[t][1] - mu_CIs[t][0])
                    for j in range(self.n_pts)
                ]
                for t in range(self.demand.T)
            ]

            sig_CIs_disc = [
                [
                    sig_CIs[t][0]
                    + (j / (self.n_pts - 1)) * (sig_CIs[t][1] - sig_CIs[t][0])
                    for j in range(self.n_pts)
                ]
                for t in range(self.demand.T)
            ]

            mu_vals = it.product(*mu_CIs_disc)
            sig_vals = it.product(*sig_CIs_disc)

            Omega = it.product(mu_vals, sig_vals)
            end = time.perf_counter()

        elif self.demand.dist in ["Poisson", "poisson"]:
            start = time.perf_counter()
            lam_0 = self.demand.omega_0
            # z = norm.ppf(1 - self.alpha / 2)
            z = np.sqrt(chi2.ppf(q=1 - self.alpha, df=self.demand.T))
            N = self.demand.N
            lam_hat = self.demand.mle
            CIs = [
                (
                    lam_hat[t] - z * np.sqrt(lam_hat[t] / N),
                    lam_hat[t] + z * np.sqrt(lam_hat[t] / N),
                )
                for t in range(self.demand.T)
            ]

            CIs_disc = [
                [
                    CIs[t][0] + (j / (self.n_pts - 1)) * (CIs[t][1] - CIs[t][0])
                    for j in range(self.n_pts)
                ]
                for t in range(self.demand.T)
            ]

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
                    if (min(omega[1]) > 0) and norm_conf_val(
                        self.demand.N, self.demand.T, omega, self.demand.mle
                    ) <= chi:
                        confset.append(omega)
                    tt = time.perf_counter() - start
                    if tt > left:
                        self.confidence_set_full = "T.O."
                        return

                self.confidence_set_full = confset
                self.time_taken += time.perf_counter() - start

        if self.demand.dist in ["Poisson", "poisson"]:
            chi = chi2.ppf(q=1 - self.alpha, df=self.demand.T)
            confset = []
            for lam in self.base_set:
                if (
                    pois_conf_val(self.demand.N, self.demand.T, lam, self.demand.mle)
                    <= chi
                ):
                    confset.append(lam)
                tt = time.perf_counter() - start
                if tt > left:
                    self.confidence_set_full = "T.O."
                    return

            self.confidence_set_full = confset
            self.time_taken += time.perf_counter() - start

        if self.demand.mle not in self.confidence_set_full:
            self.confidence_set_full.append(self.demand.mle)

    def reduce(self):
        if self.demand.dist in ["Normal", "normal"]:
            left = self.timeout - self.time_taken
            start = time.perf_counter()
            not_dominated = it.filterfalse(
                lambda x: np.any(
                    [
                        np.all(np.array(o[1]) >= np.array(x[1]))
                        for o in self.confidence_set_full
                        if o[0] == x[0] and o[1] != x[1]
                    ]
                ),
                self.confidence_set_full,
            )
            ASR = []
            for o in not_dominated:
                if time.perf_counter() - start < left:
                    ASR.append(o)
                else:
                    self.reduced = "T.O."
                    return
            end = time.perf_counter()
            self.time_taken += end - start
            self.reduced = ASR

        elif self.demand.dist in ["Poisson", "poisson"]:
            # not sure how this will work yet
            self.reduced = self.confidence_set_full

    def reduce_fast(self):
        if self.demand.dist in ["Normal", "normal"]:
            left = self.timeout - self.time_taken
            start = time.perf_counter()
            AS_conf = self.confidence_set_full.copy()
            AS_conf_ = np.array([tuple(np.array(x).flatten()) for x in AS_conf])
            sigs = list(set([o[1] for o in AS_conf]))
            sigs = [list(o) for o in sigs]
            sigs_t = [list(set([o[t] for o in sigs])) for t in range(self.demand.T)]
            sigs_t = [list(sigs_t[t]) for t in range(self.demand.T)]
            sigs_t = [sorted(x) for x in sigs_t]
            ml = max([len(sigs_t[t]) for t in range(self.demand.T)])
            sigs_t_ = np.zeros((self.demand.T, ml), dtype="float64")
            for t in range(self.demand.T):
                sigs_t_[t, : len(sigs_t[t])] = sigs_t[t]

            red_ = -1 * np.ones(AS_conf_.shape, dtype="float64")
            red = reduce_njit(AS_conf_, sigs_t_, red_)
            ASR = [
                (tuple(x[: int(len(x) / 2)]), tuple(x[int(len(x) / 2) :])) for x in red
            ]
            end = time.perf_counter()
            self.time_taken += end - start
            self.reduced = ASR

        elif self.demand.dist in ["Poisson", "poisson"]:
            # not sure how this will work yet
            self.reduced = self.confidence_set_full

    def compute_extreme_distributions(self):
        start = time.perf_counter()
        left = self.timeout - self.time_taken
        if self.demand.dist in ["normal", "Normal"]:
            if self.confidence_set_full == "T.O.":
                self.extreme_distributions = "T.O."
                return "T.O."
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
            lam_max = [
                max(lam[t] for lam in self.reduced) for t in range(self.demand.T)
            ]
            lam_min = [
                min(lam[t] for lam in self.reduced) for t in range(self.demand.T)
            ]

            self.extreme_distributions = [
                lam
                for lam in self.reduced
                if np.any(
                    np.array(
                        [
                            lam[t] in [lam_max[t], lam_min[t]]
                            for t in range(self.demand.T)
                        ]
                    )
                )
            ]
            end = time.perf_counter()
            self.time_taken += end - start
            if self.time_taken > self.timeout:
                self.extreme_distributions = "T.O."

            return
