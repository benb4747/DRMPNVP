from functions import *
from demand_RV import *
from ambiguity_set import *
from DRMPNVP import *

import numpy as np
from decimal import Decimal
from scipy.stats import norm, poisson
from math import log, exp


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
        if self.dist in ["normal", "Normal"]:
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

        if self.dist in ["poisson", "Poisson"]:
            w = np.array(self.w, dtype="float")
            days = [i for i in range(self.T) if i not in self.zero_days]
            days = np.array(days, dtype="int")
            l = self.omega # lambda

            n = len(days)
            sol = np.zeros(self.T)

            if 0 not in self.zero_days:
                pr = (
                    Decimal(self.b) - Decimal(1 + lam) * Decimal(self.w[0] - self.w[1])
                ) / Decimal(self.h + self.b)
                sol[0] = Decimal(
                    poisson.ppf(
                        q=np.float64(pr),
                        mu=l[0]
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
                            mu=sum(l[: t + 1])
                            )
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
                        poisson.ppf(
                            q=np.float64(pr),
                            mu=sum(l)
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
