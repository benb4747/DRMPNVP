import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import gurobipy as gp
from gurobipy import GRB
import time
import itertools as it
from scipy.stats import chi2
from numba import njit
from decimal import Decimal
from scipy.misc import derivative
from decimal import *
from math import log
import os, sys


def read_results(file):
    file1 = open(file, "r")
    lines = file1.readlines()
    file1.close()

    lines_new = []
    names = eval(lines[0])
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "failed" not in line:
            lines_new.append(eval(line))
    print("Dataset has %s rows." % len(lines_new))

    res_array = np.array(lines_new)
    df = pd.DataFrame(data=res_array, columns=names)
    return df


# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


@njit
def norm_conf_val(N, T, omega, omega_hat):
    val = 0
    mu, sig = omega
    mu_hat, sig_hat = omega_hat
    for t in range(T):
        val += (N / sig_hat[t] ** 2) * (mu_hat[t] - mu[t]) ** 2
        val += (2 * N / sig_hat[t] ** 2) * (sig_hat[t] - sig[t]) ** 2
    return val


def mu_sig_combos(mu_0_range, sig_0_range, T):
    return [
        (mu_0, sig_0)
        for mu_0 in list(it.combinations_with_replacement(mu_0_range, T))
        for sig_0 in list(it.combinations_with_replacement(sig_0_range, T))
    ]


class demand_RV:
    # demand_RV("normal", T, omega_0, N, index)
    def __init__(self, dist, T, omega_0, N, seed):
        self.dist = dist
        self.T = T
        self.omega_0 = omega_0
        self.seed = seed
        self.N = N

    def compute_mles(self):
        if self.dist == "normal":
            mu, sig = self.omega_0
            Sig = np.zeros((self.T, self.T))
            for t in range(self.T):
                Sig[t, t] = sig[t] ** 2
            np.random.seed(self.seed)
            samples = np.random.multivariate_normal(mean=mu, cov=Sig, size=self.N)
            mu_MLE = [np.mean(samples[:, t]) for t in range(self.T)]

            sig_MLE = [
                np.sqrt(np.mean((samples[:, t] - mu_MLE[t]) ** 2))
                for t in range(self.T)
            ]

            self.mle = (mu_MLE, sig_MLE)


class ambiguity_set:
    def __init__(self, demand, set_type, alpha, n_pts, timeout):
        self.demand = demand
        self.set_type = set_type
        self.alpha = alpha
        self.n_pts = n_pts
        self.timeout = timeout
        self.time_taken = 0

    def construct_base_set(self):
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
        self.time_taken = end - start
        if self.time_taken > self.timeout:
            self.base_set = "T.O."
        else:
            self.base_set = Omega

    def construct_confidence_set(self):
        left = self.timeout - self.time_taken
        start = time.perf_counter()
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

        if tuple(self.demand.mle) not in self.confidence_set_full:
            self.confidence_set_full.append(tuple(self.demand.mle))

    def reduce(self):
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

    def compute_extreme_distributions(self):
        start = time.perf_counter()
        left = self.timeout - self.time_taken
        if self.demand.dist == "normal":
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


class DRMPNVP:
    def __init__(self, T, W, w, p, h, b, gap, solver_timeout, solver_cores, dist, AS):
        self.T = T
        self.W = W
        self.w = w
        self.p = p
        self.h = h
        self.b = b
        self.gap = gap
        self.timeout = solver_timeout
        self.dist = dist
        self.AS = AS  # ambiguity set object
        self.solver_cores = solver_cores
        self.a = [self.h + self.b + self.p * (t == (self.T - 1)) for t in range(self.T)]

    def cost_function(self, q, omega):
        if self.dist == "normal":
            mu, sig = omega
            s = [
                np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))
                for t in range(self.T)
            ]

            alpha = [
                sum([mu[k] - q[k] for k in range(t + 1)]) / s[t] for t in range(self.T)
            ]

            obj = sum(
                [
                    (
                        (self.a[t] * norm.cdf(alpha[t]) - self.h)
                        * sum([mu[k] - q[k] for k in range(t + 1)])
                        + self.a[t] * norm.pdf(alpha[t]) * s[t]
                        + q[t] * self.w[t]
                        - self.p * mu[t]
                    )
                    for t in range(self.T)
                ]
            )
            return obj

    def nonlinear_part(self, omega, alpha, t):
        if self.dist == "normal":
            # evaluates the nonlinear part of an objective term
            # i.e.\ the non-linear part for one t.
            mu, sig = omega
            s = np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))

            return ((self.a[t] * norm.cdf(alpha) - self.h) * alpha * s) + (
                self.a[t] * norm.pdf(alpha) * s
            )

    def PWL_NLpart(self, omega, alpha, alpha_pts, t):
        ind = [
            i
            for i in range(len(alpha_pts) - 1)
            if alpha_pts[i] <= alpha and alpha <= alpha_pts[i + 1]
        ][0]
        return (
            (alpha - alpha_pts[ind + 1]) / (alpha_pts[ind] - alpha_pts[ind + 1])
        ) * self.nonlinear_part(omega, alpha_pts[ind], t) + (
            (alpha - alpha_pts[ind]) / (alpha_pts[ind + 1] - alpha_pts[ind])
        ) * self.nonlinear_part(
            omega, alpha_pts[ind + 1], t
        )

    def PWL_obj(self, q, omega, alpha_pts):
        if self.dist == "normal":
            mu, sig = omega
            s = [
                np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))
                for t in range(self.T)
            ]
            alpha = [
                sum([mu[k] - q[k] for k in range(t + 1)]) / s[t] for t in range(self.T)
            ]
            NL_part = [self.nonlinear_part(omega, alpha[t], t) for t in range(self.T)]
            obj = sum([NL_part[t] + self.w[t] * q[t] for t in range(T)])
            return obj

    def build_model(self, ambiguity_set):
        start = time.perf_counter()
        TO = False
        block_print()
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        env.setParam("NonConvex", 2)
        env.setParam("Presolve", 1)
        env.setParam("LogToConsole", 0)
        env.setParam("Threads", self.solver_cores)
        m = gp.Model(name="MPNVP_%s_day" % self.T, env=env)
        q = m.addVars([t for t in range(self.T)], vtype=GRB.CONTINUOUS, name="q")
        dummy = m.addVar(vtype=GRB.CONTINUOUS, name="dummy", lb=-GRB.INFINITY)
        enable_print()
        if self.dist == "normal":
            alpha = m.addVars(
                [(i, t) for i in range(len(ambiguity_set)) for t in range(self.T)],
                vtype=GRB.CONTINUOUS,
                name="alpha",
                lb=-GRB.INFINITY,
            )
            NL_part = m.addVars(
                [(i, t) for i in range(len(ambiguity_set)) for t in range(self.T)],
                vtype=GRB.CONTINUOUS,
                name="NL_part",
                lb=-GRB.INFINITY,
            )

            alpha_min = np.array(
                [
                    [
                        sum([mu[k] - (self.W / self.w[k]) for k in range(t + 1)])
                        / np.sqrt(sum([sig[k] ** 2 for k in range(t + 1)]))
                        for t in range(self.T)
                    ]
                    for (mu, sig) in ambiguity_set
                ]
            )
            alpha_max = np.array(
                [
                    [
                        sum([mu[k] for k in range(t + 1)])
                        / np.sqrt(sum([sig[k] ** 2 for k in range(t + 1)]))
                        for t in range(self.T)
                    ]
                    for (mu, sig) in ambiguity_set
                ]
            )
            alpha_pts = np.array(
                [
                    [
                        np.arange(
                            np.floor(alpha_min[i, t]),
                            np.ceil(alpha_max[i, t]) + self.gap,
                            self.gap,
                        )
                        for t in range(self.T)
                    ]
                    for i in range(len(ambiguity_set))
                ]
            )

            for i in range(len(ambiguity_set)):
                tt = time.perf_counter() - start
                if tt >= self.timeout:
                    TO = True
                    # print("model built in %s seconds." %np.round(end-start, 3))
                    var = [q, alpha, dummy, NL_part]
                    t_build = time.perf_counter() - start
                    return [m, var, tt, TO]

                mu, sig = ambiguity_set[i]
                s = [
                    np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))
                    for t in range(self.T)
                ]
                for t in range(self.T):
                    NL_pts = [
                        self.nonlinear_part((mu, sig), alpha, t)
                        for alpha in alpha_pts[i, t]
                    ]
                    m.addGenConstrPWL(
                        alpha[i, t],
                        NL_part[i, t],
                        alpha_pts[i, t],
                        NL_pts,
                        "NLconstr_(%s,%s)" % (i, t),
                    )
                    # NL_pts = [nonlinear_part(alpha, t, mu, sig, T, W, w, p, h, b) for alpha in alpha_pts]
                    # m.addGenConstrPWL(alpha[i, t], NL_part[i, t], alpha_pts, NL_pts, "NLconstr_(%s,%s)" %(i,t))
                    m.addConstr(
                        alpha[i, t]
                        == gp.quicksum(mu[k] - q[k] for k in range(t + 1)) / s[t],
                        name="alpha_(%s, %s)_constraint" % (i, t),
                    )

                m.addConstr(
                    dummy
                    >= gp.quicksum(
                        NL_part[i, t] + q[t] * self.w[t] - self.p * mu[t]
                        for t in range(self.T)
                    )
                )

        m.addConstr(gp.quicksum(self.w[t] * q[t] for t in range(self.T)) <= self.W)
        m.setObjective(dummy, GRB.MINIMIZE)
        end = time.perf_counter()
        # print("model built in %s seconds." %np.round(end-start, 3))

        var = [q, alpha, dummy, NL_part]
        t_build = end - start
        return [m, var, t_build, TO]

    def solve_model(self, m, var, t_build, TO, ambiguity_set):
        [q, alpha, dummy, NL_part] = var
        start = time.perf_counter()
        TO_new = t_build > self.timeout
        if TO_new:
            TO = True
            q_sol = tuple(self.T * [-1])
            worst = tuple([tuple(self.T * [-1]), tuple(self.T * [-1])])
            obj = 0
            end = time.perf_counter()
            tt = np.round(t_build + end - start, 3)
            del m
            return [q_sol, obj, worst, tt, TO]

        tt = t_build + time.perf_counter() - start
        m.Params.TimeLimit = self.timeout - tt
        m.optimize()

        TO = m.Status == GRB.TIME_LIMIT

        if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
            q_sol = np.array((m.getAttr("x", q).values()))
            obj = m.objVal
            # alpha_sol = np.array(m.getAttr("x", alpha).values()).reshape(len(self.ambiguity_set), self.T)
            NL_sol = np.array(m.getAttr("x", NL_part).values()).reshape(
                len(ambiguity_set), self.T
            )
            exp_cost_val = [self.cost_function(q_sol, o) for o in ambiguity_set]
            worst = ambiguity_set[np.argmax(exp_cost_val)]
        else:
            q_sol = tuple(self.T * [-1])
            worst = tuple([tuple(self.T * [-1]), tuple(self.T * [-1])])
            obj = 0

        end = time.perf_counter()
        tt = np.round(t_build + end - start, 3)
        # print("Model solved in %s seconds." %np.round(end-start, 3))
        del m

        return [q_sol, obj, worst, tt, TO]

    def PWL_solve(self, ambiguity_set):
        [m, var, t_build, TO] = self.build_model(ambiguity_set)
        [q_sol, obj, worst, tt, TO] = self.solve_model(
            m, var, t_build, TO, ambiguity_set
        )

        return [q_sol, obj, worst, tt, TO]

    def CS_solve(self, omega_0, verbose=False, eps=0.01):

        s = time.perf_counter()
        Omega_k = [omega_0]

        # make a set of the worst distributions
        Omega_ext = self.AS.extreme_distributions

        Q_k, T_k = [], []  # solutions and objectives
        k_max = 10
        k = 0
        reason = ""
        while k < k_max:
            # solve master problem
            tt = time.perf_counter() - s
            [m, var, t_build, TO] = self.build_model(Omega_k)
            [q, alpha, dummy, NL_part] = var
            [q_k, obj_master, o_master, tt_master, TO] = self.solve_model(
                m, var, t_build, TO, Omega_k
            )

            # calculate using self.cost_function as objective of LP is only approximate
            t_k = self.cost_function(q_k, o_master)
            C_q = [self.cost_function(q_k, omega) for omega in Omega_ext]

            worst = np.argmax(C_q)
            o_opt = Omega_ext[worst]
            C_opt = C_q[worst]

            if verbose:
                print("----- Iteration %s -----\n" % k)
                print("q = %s\n" % q_k)
                print("Master obj = %s\n" % obj_master)
                print("True obj = %s\n" % t_k)
                print("True worst-case obj = %s\n" % C_opt)
                print("")

            Q_k.append(q_k)
            T_k.append(t_k)
            repeat = list(o_opt) in Omega_k

            tt = time.perf_counter() - s
            TO_new = tt > self.timeout
            if TO or TO_new:
                reason = "TO"
                break

            if not repeat:
                Omega_k.append(list(o_opt))

            else:
                reason = "repeat"
                break

            if C_opt <= t_k + eps / 2 or repeat:
                reason = "optimal"
                break

            else:
                k += 1

        if k == k_max and reason == "":
            reason = "k"

        e = time.perf_counter()
        tt = np.round(e - s, 3)
        opt_sol = [Q_k[-1], o_opt, C_opt, tt]
        return opt_sol + [TO, reason]


class MPNVP(DRMPNVP):
    def __init__(
        self,
        T,
        W,
        w,
        p,
        h,
        b,
        gap,
        solver_timeout,
        solver_cores,
        dist,
        omega,
        budget_tol,
        tau,
        alpha_min=1e-100,
        alpha_init=0.1,
    ):

        super().__init__(
            T, W, w, p, h, b, gap, solver_timeout, solver_cores, dist, [omega]
        )

        self.budget_tol = budget_tol
        self.tau = tau
        self.alpha_min = alpha_min
        self.alpha_init = alpha_init
        self.zero_days = []
        self.omega = omega

    def KKT_solution(self, lam):
        if self.dist == "normal":
            w = np.array(self.w, dtype="float")
            days = [i for i in range(self.T) if i not in self.zero_days]
            days = np.array(days, dtype="int")
            mu, var = np.array(self.omega[0]), np.array(self.omega[1])

            n = len(days)
            sol = np.zeros(self.T)

            if 0 not in self.zero_days:
                pr = (
                    Decimal(self.b) - Decimal(1 + lam) * Decimal(self.w[0] - self.w[1])
                ) / Decimal(self.h + self.b)
                sol[0] = Decimal(
                    norm.ppf(
                        q=np.float64(pr),
                        loc=mu[0],
                        scale=np.float64(Decimal(np.sqrt(var[0]))),
                    )
                )
                if not np.isfinite(sol[0]):
                    return sol
            for t in range(1, self.T - 1):
                if t not in self.zero_days:

                    pr = (
                        Decimal(self.b)
                        - Decimal(1 + lam) * Decimal(self.w[t] - self.w[t + 1])
                    ) / Decimal(self.h + self.b)

                    sol[t] = Decimal(
                        norm.ppf(
                            q=np.float64(pr),
                            loc=sum(mu[: t + 1]),
                            scale=np.float64(Decimal(np.sqrt(sum(var[: t + 1])))),
                        )
                    ) - Decimal(sum(sol[:t]))
                    if not np.isfinite(sol[t]):
                        return sol
                    # print(Decimal(sum(sol[:t])))
                    # - int(sol[days[i] - 1] > 0) * norm.ppf(q=(b  - (1 + lam) *(w[days[i-1]] - w[days[i]])) / (h + b),
                    # loc=sum(mu[days[:i]]), scale=np.sqrt(sum(var[days[:i]]))
            if self.T - 1 not in self.zero_days:
                pr = (
                    Decimal(self.b)
                    - Decimal(1 + lam) * Decimal(self.w[-1])
                    + Decimal(self.p)
                ) / Decimal(self.h + self.b + self.p)
                sol[self.T - 1] = (
                    Decimal(
                        norm.ppf(
                            q=np.float64(pr),
                            loc=sum(mu),
                            scale=np.float64(Decimal(np.sqrt(sum(var)))),
                        )
                    )
                    # - int(sol[- 2] > 0) * norm.ppf(q=(b - (1 + lam) *(w[days[-2]] - w[days[-1]])) / (h + b),
                    # loc=sum(mu[:days[n-1]]), scale=np.sqrt(sum(var[:days[n-1]]))
                    - Decimal(sum(sol[:-1]))
                )

            return sol

    def alfares_solve(self):
        self.w = np.array(self.w, dtype="float")
        [self.h, self.b, self.p, self.W] = [
            float(x) for x in [self.h, self.b, self.p, self.W]
        ]
        days = list(range(self.T))
        opt = False
        zero_days = []
        constraint_binding = False
        power = int(-log(self.alpha_min) / log(10))
        getcontext().prec = power

        lam = Decimal(0)
        # print("Precision = %s." %getcontext().prec)
        while not opt:
            # print("Solving with days %s." %days)
            if constraint_binding and len(days) == 1:
                q = np.zeros(self.T)
                q[days[0]] = Decimal(self.W) / Decimal(self.w[days[0]])
            else:
                q = self.KKT_solution(lam=lam)
            for i in range(self.T):
                if q[i] < 0:
                    q[i] = 0

            if sum(self.w * q) - self.W > self.budget_tol:
                if days == list(range(self.T)) and sum(self.w * q) > self.W:
                    constraint_binding = True
                    # constraint is binding because initial sol
                    # is not feasible

                # lambda needs to be increased to bring orders down
                a = 0
                lam_UBs = [
                    Decimal(self.b) / Decimal(self.w[i] - self.w[i + 1]) - Decimal(1)
                    for i in range(self.T - 1)
                ] + [
                    (Decimal(self.b) + Decimal(self.p)) / Decimal(self.w[-1])
                    - Decimal(1)
                ]
                b_ = min([lam_UBs[i] for i in days])

                alpha = Decimal(self.alpha_init)
                alpha_min = Decimal(self.alpha_min)
                budget_tol = Decimal(self.budget_tol)
                # lam = Decimal(a)
                lam_next = lam
                q_temp = self.KKT_solution(lam=lam)
                q_next = q_temp
                done = False
                k = 0
                while not done:
                    # this loop runs until alpha is as low as it can be and we have the first
                    # occurence of negative/nan/optimal solution
                    lam_old = lam
                    while lam_old + alpha >= Decimal(b_):
                        alpha = Decimal(alpha) * Decimal(self.tau)
                        alpha_min = min(alpha, self.alpha_min)

                    lam_next = lam_old + alpha
                    q_next = self.KKT_solution(lam=lam_next)

                    if lam_next == lam_old or (
                        alpha == self.alpha_min and np.all(q_next == q_temp)
                    ):
                        break

                    if (
                        (
                            len(np.where(q_next < 0)[0]) == 1
                            and np.all(np.isfinite(q_next))
                            and Decimal(sum(q_next * self.w)) - Decimal(self.W) > 0
                        )
                        or Decimal(sum(q_next * self.w)) - Decimal(self.W) > 0
                        or alpha == self.alpha_min
                    ):
                        lam = lam_next
                        q_temp = q_next

                    else:
                        # if we are skipping it then we reduce alpha
                        alpha_next = Decimal(self.tau) * Decimal(alpha)
                        if (
                            alpha_next <= self.alpha_min
                            or lam_old + alpha_next == lam_old
                        ):
                            # this catches cases where alpha gets so small
                            # that it becomes zero
                            alpha_next = alpha
                            self.alpha_min = alpha

                        alpha = alpha_next
                        q_next = q_temp

                    if (
                        abs(Decimal(self.W) - Decimal(sum(q_next * self.w)))
                        <= self.budget_tol
                        and np.all(q_next >= 0)
                    ) or (np.any(q_next < 0)):
                        # solution is optimal
                        done = True

                q = q_next
                # print(" - New sol: %s. Budget used: %s." %(q, sum(w * q)))

                if np.any(np.isnan(q)):
                    break

                elif np.any(q < 0) or Decimal(sum(self.w * q)) - Decimal(
                    self.W
                ) > Decimal(self.budget_tol):
                    LBs = []
                    q_LBs = []
                    for day in days:
                        self.zero_days.append(day)
                        days_temp = [
                            t for t in range(self.T) if t not in self.zero_days
                        ]
                        if len(days_temp) > 1:
                            # compute the solution resulting from setting this day
                            # to zero and leaving other parameters the same.
                            # use the current lambda because lambda will need to be
                            # at least this value in order to reach optimality.
                            q_LB = self.KKT_solution(lam=lam)
                            q_LBs.append(q_LB)
                        else:
                            day_ = days_temp[0]
                            q_LB = np.zeros(self.T)
                            q_LB[day_] = self.W / self.w[day_]
                        cost_LB = self.cost_function(q_LB, self.omega)
                        LBs.append(cost_LB)

                        self.zero_days.remove(day)
                    # print("days: %s" %days)
                    # print("q_LB: %s" %q_LBs)
                    # print("LBs: %s" %LBs)
                    day = days[np.argmin(LBs)]
                    # print("Removing day %s. " %day)
                    days.remove(day)
                    self.zero_days.append(day)

                    if days == []:
                        opt = True

                elif sum(self.w * q) <= Decimal(self.W) + Decimal(self.budget_tol):
                    opt = True

            else:
                opt = True
                # print("Optimal q is: %s." %q)
        self.alfares_sol = q

        return q
