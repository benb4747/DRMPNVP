from multiprocessing import Pool
import sys, os
from src.ambiguity_set import *
from src.demand_RV import *
from src.DRMPNVP import *
from src.MPNVP import *

import numpy as np
from decimal import Decimal, getcontext
from scipy.stats import norm, poisson
from math import log, exp


def test_algorithms(inp):
    (
        index,
        rep,
        num_reps,
        mu_0,
        sig_0,
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        n_pts,
        CS_tol,
        timeout,
        solver_cores,
        N,
        alpha,
    ) = inp

    headers = [
        index,
        rep,
        num_reps,
        mu_0,
        sig_0,
        T,
        W,
        tuple(w),
        p,
        h,
        b,
        PWL_gap,
        n_pts,
        CS_tol,
        timeout,
        N,
        alpha,
    ]

    dist = "normal"
    omega_0 = (mu_0, sig_0)
    # construct MLEs and confidence set
    X = demand_RV(dist, T, (mu_0, sig_0), N, seed=index * num_reps + rep)
    X.compute_mles()
    MLE_neg = np.any(np.array(X.mle) < 0)
    if MLE_neg:
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write("Input %s had negative MLEs. \n" % index)
            return
    AS = ambiguity_set(X, "confidence_set", alpha, n_pts, timeout)
    s = time.perf_counter()
    try:
        AS.construct_base_set()
    except MemoryError:
        AS.base_set = "T.O."
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s had MemoryError while constructing base AS. \n" % index
                )
            return
    e = time.perf_counter()
    t_base = np.round(e - s, 5)

    if AS.base_set == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s timed out while constructing base AS. \n" % index
                )
            return
    s = time.perf_counter()
    AS.construct_confidence_set()
    e = time.perf_counter()
    t_conf = np.round(e - s, 5)
    if AS.confidence_set_full == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s timed out while constructing confidence set. \n" % index
                )
            return
    #     AS.reduce()
    #     if AS.reduced == "T.O.":
    #         for f in [count_file, results_file]:
    #             with open(f, "a") as myfile:
    #                 myfile.write("Input %s timed out while reducing set. \n" % index)
    #             return
    AS.reduced = AS.confidence_set_full
    s = time.perf_counter()
    AS.compute_extreme_distributions()
    e = time.perf_counter()
    t_ext = np.round(e - s, 5)
    if AS.extreme_distributions == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s timed out while constructing extreme set. \n" % index
                )
            return

    num_dists = len(AS.reduced)
    if num_dists <= 1:
        with open(count_file, "a") as myfile:
            myfile.write(
                "Input %s only had %s distribution. Skipping! \n" % (index, num_dists)
            )
        return

    t_AS = np.round(t_base + t_conf + t_ext, 5)
    headers += [
        len(AS.confidence_set_full),
        len(AS.reduced),
        t_base,
        t_conf,
        t_ext,
        t_AS,
    ]
    DRO_problem = DRMPNVP(T, W, w, p, h, b, PWL_gap, timeout, solver_cores, dist, AS)

    PWL_sol = DRO_problem.PWL_solve(AS.reduced)
    PWL_q, PWL_obj, PWL_worst, PWL_tt, PWL_TO = PWL_sol
    if PWL_q[0] != -1:
        PWL_obj = DRO_problem.cost_function(PWL_q, PWL_worst)
        PWL_true_worst, PWL_true_worst_obj = DRO_problem.DSP(PWL_q, N, X.mle)
        PWL_true_obj = np.round(DRO_problem.cost_function(PWL_q, omega_0), 5)
        PWL_tt = PWL_tt + t_AS - t_ext
    else:
        PWL_obj = -1
        PWL_true_worst = PWL_worst
        PWL_true_worst_obj = -1
        PWL_true_obj = -1

    CS_sol = DRO_problem.CS_solve(AS.reduced[0])
    CS_q, CS_worst, CS_obj, CS_tt, CS_TO, CS_reason = CS_sol
    if CS_q[0] != -1:
        CS_obj = DRO_problem.cost_function(CS_q, CS_worst)
        CS_true_worst, CS_true_worst_obj = DRO_problem.DSP(CS_q, N, X.mle)
        CS_true_obj = np.round(DRO_problem.cost_function(CS_q, omega_0), 5)
        CS_tt = np.round(CS_tt + t_AS, 5)
    else:
        CS_obj = -1
        CS_true_worst = CS_worst
        CS_true_worst_obj = -1
        CS_true_obj = -1

    CSO_sol = DRO_problem.CS_solve(
        omega_0, MLE=X.mle, N=N, discrete=False, verbose=False, eps=1e-6
    )
    CSO_q, CSO_worst, CSO_obj, CSO_tt, CSO_TO, CSO_reason = CSO_sol
    if CSO_q[0] != -1:
        CSO_obj = DRO_problem.cost_function(CSO_q, CSO_worst)
        CSO_true_worst, CSO_true_worst_obj = DRO_problem.DSP(CSO_q, N, X.mle)
        CSO_true_obj = np.round(DRO_problem.cost_function(CSO_q, omega_0), 5)
    else:
        CSO_obj = -1
        CSO_true_worst = CSO_worst
        CSO_true_worst_obj = -1
        CSO_true_obj = -1

    CS_reasons = ["TO", "repeat", "optimal", "k"]
    CS_reason = CS_reasons.index(CS_reason)
    CSO_reason = CS_reasons.index(CSO_reason)

    MLE_problem = MPNVP(
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        timeout,
        solver_cores,
        dist,
        X.mle,
        budget_tol=1e-6,
        tau=0.5,
        alpha_min=1e-100,
        alpha_init=0.1,
    )
    s = time.perf_counter()
    MLE_q, MLE_obj = MLE_problem.SLSQP_solve()
    e = time.perf_counter()
    MLE_tt = np.round(e - s, 3)
    MLE_obj = MLE_problem.cost_function(MLE_q, X.mle)
    MLE_true_obj = MLE_problem.cost_function(MLE_q, omega_0)

    omega0_problem = MPNVP(
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        timeout,
        solver_cores,
        dist,
        omega_0,
        budget_tol=1e-6,
        tau=0.5,
        alpha_min=1e-100,
        alpha_init=0.1,
    )
    s = time.perf_counter()
    omega0_q, omega0_obj = omega0_problem.SLSQP_solve()
    e = time.perf_counter()
    omega0_tt = np.round(e - s, 3)

    s = time.perf_counter()
    mb_q, mb_obj = DRO_problem.moment_based(X.mle)
    e = time.perf_counter()
    mb_true_obj = np.round(MLE_problem.cost_function(mb_q, omega_0), 5)
    mb_tt = np.round(e - s, 3)
    s = time.perf_counter()
    saa_q, saa_obj = DRO_problem.SAA(N, seed=index * num_reps + rep, omega=X.mle)
    e = time.perf_counter()
    saa_tt = np.round(e - s, 3)
    saa_true_obj = np.round(MLE_problem.cost_function(saa_q, omega_0), 5)

    with open(count_file, "a") as myfile:
        myfile.write("Finished solving input %s replication %s. \n" % (index, rep))

    res_list = headers + [
        tuple(np.round(PWL_q, 3)),
        tuple([tuple(i) for i in np.round(PWL_worst, 3)]),
        np.round(PWL_obj, 5),
        tuple([tuple(i) for i in np.round(PWL_true_worst, 3)]),
        np.round(PWL_true_worst_obj, 5),
        PWL_tt,
        int(PWL_TO),
        PWL_true_obj,
        tuple(np.round(CS_q, 3)),
        tuple([tuple(i) for i in np.round(CS_worst, 3)]),
        np.round(CS_obj, 5),
        tuple([tuple(i) for i in np.round(CS_true_worst, 3)]),
        np.round(CS_true_worst_obj, 5),
        CS_tt,
        int(CS_TO),
        CS_reason,
        CS_true_obj,
        tuple(np.round(CSO_q, 3)),
        tuple([tuple(i) for i in np.round(CSO_worst, 3)]),
        np.round(CSO_obj, 5),
        tuple([tuple(i) for i in np.round(CSO_true_worst, 3)]),
        np.round(CSO_true_worst_obj, 5),
        CSO_tt,
        int(CSO_TO),
        CSO_reason,
        CSO_true_obj,
        tuple(np.round(MLE_q, 3)),
        tuple([tuple(i) for i in np.round(X.mle, 3)]),
        np.round(MLE_obj, 5),
        MLE_tt,
        MLE_true_obj,
        tuple(np.round(mb_q, 3)),
        np.round(mb_obj, 5),
        mb_tt,
        mb_true_obj,
        tuple(np.round(saa_q, 3)),
        np.round(saa_obj, 5),
        saa_true_obj,
        saa_tt,
        MLE_neg,
        np.round(omega0_obj, 5),
        tuple(np.round(omega0_q, 3)),
        np.round(omega0_tt, 3),
    ]

    with open(results_file, "a") as res_file:
        res_file.write(str(res_list) + "\n")

    return res_list


import logging


def test_algorithms_mp(inp):
    ind, rep = inp[0], inp[1]
    try:
        return test_algorithms(inp)
    except Exception:
        with open(results_file, "a") as res_file:
            res_file.write("Input %s failed on replication %s.\n" % (ind, rep))
        logging.exception("Input %s failed on replication %s.\n" % (ind, rep))


# T = int(sys.argv[1]) + 1
num_processors = 20
gurobi_cores = 4
loop_cores = int(num_processors / gurobi_cores)
timeout = 4 * 60 * 60

T_vals = list(range(2, 5)) + [10]
# mu_0_range = range(3, 40)
# sig_0_range = range(3, 15)
mu_0_range = range(1, 21)
sig_0_range = range(1, 11)
num_omega0 = 3

PWL_gap_vals = list(reversed([0.1, 0.25, 0.5]))
disc_pts_vals = [3, 5, 10]
# M = disc_pts_vals[int(sys.argv[1]) - 1]
p_range = list(100 * np.array(range(1, 3)))
h_range = list(100 * np.array(range(1, 3)))
b_range = list(100 * np.array(range(1, 3)))
W_range = [4000, 4000, 8000, 20000]
N_vals = [10, 25, 50]

gap = PWL_gap_vals[int(sys.argv[1]) - 1]

omega0_all = [
    [
        (m, s)
        for (m, s) in mu_sig_combos(mu_0_range, sig_0_range, T_)
        if np.all(np.array([s[t] <= m[t] / 3 for t in range(T_)]))
    ]
    for T_ in T_vals
    if T_ < 10
]

# omega0_all = [mu_sig_combos(mu_0_range, sig_0_range, T_) for T_ in T_vals]

omega0_vals = []
for T_ in T_vals:
    if T_ < 10:
        np.random.seed(T_vals.index(T_))
        indices = np.random.choice(range(len(omega0_all[T_vals.index(T_)])), num_omega0)
        omega0_vals.append([omega0_all[T_vals.index(T_)][i] for i in indices])
    else:
        mu0_ = range(9, 21)
        lol = []
        for n in range(num_omega0):
            np.random.seed(T_vals.index(T_) * n)
            mu0 = np.random.choice(mu0_, T_)
            sig0 = []
            for t in range(T_):
                np.random.seed(T_vals.index(T_) * n * t)
                sig0.append(np.random.choice(range(1, int(mu0[t] / 3))))
            lol.append((tuple(mu0), tuple(sig0)))
        omega0_vals = omega0_vals + [lol]


inputs = [
    (
        mu_0,
        sig_0,
        T_,
        W_range[T_vals.index(T_)],
        w,
        p,
        h,
        b,
        PWL_gap,
        n_pts,
        0.001,
        timeout,
        gurobi_cores,
        N,
        0.05,
    )
    for T_ in T_vals
    for (mu_0, sig_0) in omega0_vals[T_vals.index(T_)]
    for p in p_range
    for h in h_range
    for b in b_range
    for PWL_gap in PWL_gap_vals
    for N in N_vals
    for n_pts in disc_pts_vals
    for w in [list(100 * np.array(range(1, T_ + 1))[::-1])]
    if b > max([w[t] - w[t + 1] for t in range(T_ - 1)])
]

for i in range(len(inputs)):
    inputs[i] = tuple([i] + list(inputs[i]))
num_inputs = len(inputs)

inps = inputs
repeated_inputs = [tuple([i[0], 0, 1] + list(i)[1:]) for i in inps]
inputs = repeated_inputs
names = [
    "ind",
    "rep",
    "num_reps",
    "mu_0",
    "sig_0",
    "T",
    "W",
    "w",
    "p",
    "h",
    "b",
    "PWL_gap",
    "n_pts",
    "CS_tol",
    "timeout",
    "N",
    "alpha",
    "num_dists_full",
    "num_dists_reduced",
    "t_base",
    "t_conf",
    "t_ext",
    "t_AS",
    "PWL_q",
    "PWL_worst",
    "PWL_obj",
    "PWL_true_worst",
    "PWL_true_worst_obj",
    "PWL_tt",
    "PWL_TO",
    "PWL_true_obj",
    "CS_q",
    "CS_worst",
    "CS_obj",
    "CS_true_worst",
    "CS_true_worst_obj",
    "CS_tt",
    "CS_TO",
    "CS_reason",
    "CS_true_obj",
    "CSO_q",
    "CSO_worst",
    "CSO_obj",
    "CSO_true_worst",
    "CSO_true_worst_obj",
    "CSO_tt",
    "CSO_TO",
    "CSO_reason",
    "CSO_true_obj",
    "MLE_q",
    "MLE_worst",
    "MLE_obj",
    "MLE_tt",
    "MLE_true_obj",
    "mb_q",
    "mb_obj",
    "mb_tt",
    "mb_true_obj",
    "saa_q",
    "saa_obj",
    "saa_true_obj",
    "saa_tt",
    "MLE_neg",
    "omega0_obj",
    "omega0_q",
    "omega0_tt",
]

results_file = "results_normal_new_%s.txt" % int(sys.argv[1])
count_file = "count_normal_new.txt"
test_full = inputs

res_all = "normal_new.txt"

continuing = False
just10 = True

if continuing:
    file1 = open(res_all, "r")
    lines = file1.readlines()
    file1.close()

    lines_new = []
    failed = []
    for line in lines:
        line = line.rstrip("\n")
        if "failed" in line:
            i = eval(line[line.index(" ") + 1 : line.index("f") - 1])
            r = eval(line[line.index(".") - 2 : line.index(".")])
            failed.append((i, r))
        else:
            if type(eval(line)[0]) == str:
                names = eval(line)
            else:
                lines_new.append(eval(line))

    res_array = np.array(lines_new)
    df = pd.DataFrame(data=res_array, columns=names)

    file1 = open(count_file, "r")
    lines = file1.readlines()
    file1.close()

    timed_out = []
    failed = []
    for line in lines:
        line = line.rstrip("\n")
        if "negative" in line:
            i = int(line[line.index("t") + 2 : line.index("h") - 1])
            r = 0
            failed.append((i, r))
        elif "timed" in line:
            i = [int(i) for i in line.split() if i.isdigit()][0]
            timed_out.append((i, 0))

    done = list(zip(df.ind, df.rep)) + failed + timed_out

    not_done = [i for i in test_full if (i[0], i[1]) not in done]

    test = not_done
else:
    test = test_full

# if T == 2 and continuing:
if int(sys.argv[1]) == 1 and continuing:
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving the %s instances that didn't finish solving before. \n"
            % len(test)
        )

test_ = [i for i in test if i[names.index("PWL_gap")] == gap]
if just10:
    test_ = [i for i in test_ if i[names.index("T")] == 10]

# wipes results file
# if T == 2 and not continuing:
if int(sys.argv[1]) == 1 and not continuing:
    open(count_file, "w").close()
    open(results_file, "w").close()
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving %s instances with the repeated sampling approach. \n"
            % len(test)
        )
    if not just10:
        with open(results_file, "a") as myfile:
            myfile.write(str(names) + "\n")

# res = [test_algorithms(inp) for inp in test]

if __name__ == "__main__":
    with Pool(processes=loop_cores) as p:
        res = list(p.imap(test_algorithms_mp, test_))
