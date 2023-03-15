# MLE script
from multiprocessing import Pool
import sys, os
from src.ambiguity_set import *
from src.demand_RV import *
from src.DRMPNVP import *
from src.MPNVP import *


def test_algorithms(inp):
    (
        index,
        rep,
        num_MLE,
        lam_0,
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        n_pts,
        CS_tol,
        solver_timeout,
        solver_cores,
        N,
        alpha,
    ) = inp

    headers = [
        index,
        rep,
        num_MLE,
        lam_0,
        T,
        W,
        tuple(w),
        p,
        h,
        b,
        PWL_gap,
        n_pts,
        CS_tol,
        solver_timeout,
        N,
        alpha,
    ]

    dist = "Poisson"
    # construct MLEs and confidence set
    X = demand_RV(dist, T, lam_0, N, seed=index * num_reps + rep)
    X.compute_mles()
    MLE_neg = np.any(np.array(X.mle) < 0)
    if MLE_neg:
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write("Input %s had negative MLEs. \n" % index)
            return
    headers += [1, 1]
    DRO_problem = DRMPNVP(
        T, W, w, p, h, b, PWL_gap, solver_timeout, solver_cores, dist, AS=[]
    )

    # PWL solution for MLE distribution
    PWL_MLE_sol = DRO_problem.PWL_solve([X.mle])
    PWL_MLE_q, PWL_MLE_obj, PWL_MLE_worst, PWL_MLE_tt, PWL_MLE_TO = PWL_MLE_sol
    PWL_MLE_obj = DRO_problem.cost_function(PWL_MLE_q, PWL_MLE_worst)
    PWL_MLE_true_obj = np.round(DRO_problem.cost_function(PWL_MLE_q, lam_0), 5)

    # PWL solution for lam_0
    PWL_lam0_sol = DRO_problem.PWL_solve([lam_0])
    (
        PWL_lam0_q,
        PWL_lam0_obj,
        PWL_lam0_worst,
        PWL_lam0_tt,
        PWL_lam0_TO,
    ) = PWL_lam0_sol
    PWL_lam0_obj = DRO_problem.cost_function(PWL_lam0_q, PWL_lam0_worst)
    PWL_lam0_true_obj = np.round(DRO_problem.cost_function(PWL_lam0_q, lam_0), 5)

    # Alfares solution for MLE distribution
    fd_MLE_problem = MPNVP(
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        solver_timeout,
        solver_cores,
        dist,
        X.mle,
        budget_tol=1e-6,
        tau=0.5,
        alpha_min=1e-100,
        alpha_init=0.1,
    )
    s = time.perf_counter()
    fd_MLE_q = fd_MLE_problem.alfares_solve()
    e = time.perf_counter()
    fd_MLE_tt = np.round(e - s, 3)
    fd_MLE_obj = fd_MLE_problem.cost_function(fd_MLE_q, X.mle)
    fd_MLE_true_obj = np.round(fd_MLE_problem.cost_function(fd_MLE_q, lam_0), 5)

    # Alfares solution for lam_0
    fd_lam0_problem = MPNVP(
        T,
        W,
        w,
        p,
        h,
        b,
        PWL_gap,
        solver_timeout,
        solver_cores,
        dist,
        lam_0,
        budget_tol=1e-6,
        tau=0.5,
        alpha_min=1e-100,
        alpha_init=0.1,
    )
    s = time.perf_counter()
    fd_lam0_q = fd_lam0_problem.alfares_solve()
    e = time.perf_counter()
    fd_lam0_tt = np.round(e - s, 3)
    fd_lam0_obj = fd_lam0_problem.cost_function(fd_lam0_q, lam_0)
    fd_lam0_true_obj = np.round(fd_lam0_problem.cost_function(fd_lam0_q, lam_0), 5)

    with open(count_file, "a") as myfile:
        myfile.write("Finished solving input %s replication %s. \n" % (index, rep))

    res_list = headers + [
        tuple(np.round(PWL_lam0_q, 3)),
        tuple(np.round(PWL_lam0_worst, 3)),
        np.round(PWL_lam0_obj, 5),
        PWL_lam0_tt,
        int(PWL_lam0_TO),
        PWL_lam0_true_obj,
        tuple(np.round(PWL_MLE_q, 3)),
        tuple(np.round(PWL_MLE_worst, 3)),
        np.round(PWL_MLE_obj, 5),
        PWL_MLE_tt,
        int(PWL_MLE_TO),
        PWL_MLE_true_obj,
        tuple(np.round(fd_lam0_q, 3)),
        tuple(np.round(X.mle, 3)),
        np.round(fd_lam0_obj, 5),
        fd_lam0_tt,
        fd_lam0_true_obj,
        tuple(np.round(fd_MLE_q, 3)),
        tuple(np.round(X.mle, 3)),
        np.round(fd_MLE_obj, 5),
        fd_MLE_tt,
        fd_MLE_true_obj,
        MLE_neg,
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


num_processors = 32
gurobi_cores = 4
loop_cores = int(num_processors / gurobi_cores)
timeout = 4 * 60 * 60

T_vals = range(2, 5)

# PWL_gap_vals = list(reversed([0.1, 0.25, 0.5]))
PWL_gap_vals = [1]
disc_pts_vals = [3, 5, 10]
p_range = list(100 * np.array(range(1, 3)))
h_range = list(100 * np.array(range(1, 3)))
b_range = list(100 * np.array(range(1, 3)))
W_range = [4000, 4000, 8000]
N_vals = [10, 25, 50]

with open("means.txt", "r") as f:
    lines = f.readlines()

lam0_vals = [[eval(mu) for mu in lines if len(eval(mu)) == T] for T in T_vals]

lam_0_range = list(range(5, 20))
for T in T_vals:
    lam0_all = list(it.product(*[lam_0_range for t in range(T)]))
    np.random.seed(2022)
    lam0_ind = np.random.choice(range(len(lam0_all)), 6)
    lam0_vals[T_vals.index(T)] += [
        lam0_all[i] for i in lam0_ind if lam0_all[i] not in lam0_vals[T_vals.index(T)]
    ]

inputs = [
    (
        lam_0,
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
    for lam_0 in lam0_vals[T_vals.index(T_)]
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

names = [
    "ind",
    "rep",
    "num_MLE",
    "lam_0",
    "T",
    "W",
    "w",
    "p",
    "h",
    "b",
    "PWL_gap",
    "n_pts",
    "CS_tol",
    "solver_timeout",
    "N",
    "alpha",
    "num_dists_full",
    "num_dists_reduced",
    "PWL_lam0_q",
    "PWL_lam0_worst",
    "PWL_lam0_obj",
    "PWL_lam0_tt",
    "PWL_lam0_TO",
    "PWL_lam0_true_obj",
    "PWL_MLE_q",
    "PWL_MLE_worst",
    "PWL_MLE_obj",
    "PWL_MLE_tt",
    "PWL_MLE_TO",
    "PWL_MLE_true_obj",
    "fd_lam0_q",
    "fd_lam0_worst",
    "fd_lam0_obj",
    "fd_lam0_tt",
    "fd_lam0_true_obj",
    "fd_MLE_q",
    "fd_MLE_worst",
    "fd_MLE_obj",
    "fd_MLE_tt",
    "fd_MLE_true_obj",
    "MLE_neg",
]

num_reps = 1
num_instances = "all"

if num_reps > 1:
    results_file = "results_MLE_poisson.txt"
    count_file = "count_MLE_poisson.txt"
    repeated_inputs = []
    counter = 0
    for T_ in T_vals:
        if type(num_instances) == str:
            inps = [i for i in inputs if i[3] == T_]
        else:
            inps = [i for i in inputs if i[3] == T_][:num_instances]
        for rep in range(num_reps):
            repeated_inputs += [
                tuple([i[0], rep, num_reps] + list(i)[1:]) for i in inps
            ]
else:
    results_file = "results_MLE_poisson_gap.txt"
    count_file = "count_MLE_poisson_gap.txt"
    inps = inputs
    repeated_inputs = [tuple([i[0], 0, 1] + list(i)[1:]) for i in inps]

inputs = repeated_inputs

# test_full = [
#   i for i in inputs if (i[names.index("T")], i[names.index("n_pts")]) != (4, 10)
#  and (i[names.index("T")], i[names.index("n_pts")]) != (3, 10)
# ]

test_full = inputs

continuing = False

if continuing:
    file1 = open(results_file, "r")
    lines = file1.readlines()
    file1.close()

    lines_new = []
    failed = []
    names = eval(lines[0])
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "failed" not in line:
            lines_new.append(eval(line))
        else:
            i = eval(line[line.index(" ") + 1 : line.index("f") - 1])
            r = eval(line[line.index(".") - 2 : line.index(".")])
            failed.append((i, r))

    res_array = np.array(lines_new)
    df = pd.DataFrame(data=res_array, columns=names)

    done = list(zip(df.ind, df.rep)) + failed
    not_done = [i for i in test_full if (i[0], i[1]) not in done]

    test = not_done
else:
    test = test_full

if continuing:
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving the %s instances that didn't solve before. \n"
            % len(test)
        )

# test = [i for i in test if i[3] == T][:32]


# wipes results file
if not continuing:
    open(count_file, "w").close()
    open(results_file, "w").close()
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving %s instances with the repeated sampling approach. \n"
            % len(repeated_inputs)
        )
    with open(results_file, "a") as myfile:
        myfile.write(str(names) + "\n")


if __name__ == "__main__":
    with Pool(processes=loop_cores) as p:
        res = list(p.imap(test_algorithms_mp, test))
