from functions import *
from demand_RV import *
from ambiguity_set import *
import numpy as np
from scipy.stats import chi2, norm, poisson
import gurobipy as gp
from gurobipy import GRB
import time


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
        if self.dist in ["normal", "Normal"]:
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

        elif self.dist in ["Poisson", "poisson"]:
            lam = omega
            Q = [sum(q[:t+1]) for t in range(self.T)]
            Lam = [sum(lam[:t+1]) for t in range(self.T)]
            obj = sum([
                Q[t] * self.a[t] * poisson.cdf(Q[t], Lam[t]) 
                - Lam[t] * self.a[t] * poisson.cdf(Q[t] - 1, Lam[t])
                + (self.b + self.p * int(t == self.T - 1)) * (Lam[t] - Q[t]) 
                + q[t] * self.w[t] - self.p * lam[t] 
                for t in range(self.T)
            ])
            return obj

    def nonlinear_part(self, omega, alpha, t):
        if self.dist in ["normal", "Normal"]:
            # evaluates the nonlinear part of an objective term
            # i.e.\ the non-linear part for one t.
            mu, sig = omega
            s = np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))

            return ((self.a[t] * norm.cdf(alpha) - self.h) * alpha * s) + (
                self.a[t] * norm.pdf(alpha) * s
            )

        elif self.dist in ["Poisson", "poisson"]:
            Lam = omega
            Q = alpha
            return (self.a[t] * Q * poisson.cdf(Q, Lam) 
                    - Lam * self.a[t] * poisson.cdf(Q - 1, Lam))

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
        if self.dist in ["normal", "Normal"]:
            mu, sig = omega
            s = [
                np.sqrt(np.sum([sig[k] ** 2 for k in range(t + 1)]))
                for t in range(self.T)
            ]
            alpha = [
                sum([mu[k] - q[k] for k in range(t + 1)]) / s[t] for t in range(self.T)
            ]
            PWL_NL_part = [self.PWL_NLpart(omega, alpha[t], alpha_pts, t) for t in range(self.T)]
            obj = sum([PWL_NL_part[t] + self.w[t] * q[t] for t in range(self.T)])
            return obj
        
        elif self.dist in ["poisson", "Poisson"]:
            lam = omega
            Q = [sum([q[k] for k in range(t+1)]) for t in range(self.T)]
            Lam = [sum([lam[k] for k in range(t+1)]) for t in range(self.T)]
            NL_part = [self.PWL_NLpart(Lam[t], Q[t], alpha_pts[t], t) for t in range(self.T) ]
            obj = sum( [NL_part[t] + (self.a[t] - self.h) * (Lam[t] - Q[t]) 
                        + self.w[t] * q[t] - self.p * lam[t] for t in range(self.T)] )
                   
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
        NL_part = m.addVars(
            [(i, t) for i in range(len(ambiguity_set)) for t in range(self.T)],
            vtype=GRB.CONTINUOUS,
            name="NL_part",
            lb=-GRB.INFINITY,
        )
        enable_print()
        if self.dist in ["normal", "Normal"]:
            alpha = m.addVars(
                [(i, t) for i in range(len(ambiguity_set)) for t in range(self.T)],
                vtype=GRB.CONTINUOUS,
                name="alpha",
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

                var = [q, alpha, dummy, NL_part]

        elif self.dist in ["Poisson", "poisson"]:
            Q = m.addVars(range(self.T))
            m.addConstrs(Q[t] == gp.quicksum(q[k] for k in range(t+1))
                        for t in range(self.T))
            
            Q_pts = [np.arange(0, sum([self.W / self.w[k] for k in range(t+1)]), self.gap)
                    for t in range(self.T)]
            for i in range(len(ambiguity_set)):
                tt = time.perf_counter() - start
                if tt >= self.timeout:
                    TO = True
                    q_sol = tuple(self.T * [-1])
                    worst = tuple([tuple(self.T * [-1]), tuple(self.T * [-1])])
                    obj = 0
                    del m
                    return [np.round(q_sol, 3), worst, np.round(obj,3), np.round(tt, 3), TO]
                
                lam = ambiguity_set[i]
                Lam = [sum(lam[:t+1]) for t in range(self.T)]
                for t in range(self.T):
                    NL_pts = [self.nonlinear_part(Lam[t], Q_, t) for Q_ in Q_pts[t]]
                    m.addGenConstrPWL(Q[t], NL_part[i, t], Q_pts[t], NL_pts, "NLconstr_(%s,%s)" %(i,t))

                m.addConstr(dummy >= gp.quicksum(NL_part[i, t] + (self.a[t] - self.h) * (Lam[t] - Q[t])
                                                + q[t] * self.w[t] - self.p * lam[t] 
                                                for t in range(self.T)))
                var = [q, Q, dummy, NL_part]

        m.addConstr(gp.quicksum(self.w[t] * q[t] for t in range(self.T)) <= self.W)
        m.setObjective(dummy, GRB.MINIMIZE)
        end = time.perf_counter()
        # print("model built in %s seconds." %np.round(end-start, 3))

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
