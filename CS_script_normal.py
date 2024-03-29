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

    if AS.base_set == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s timed out while constructing base AS. \n" % index
                )
            return
    AS.construct_confidence_set()
    if AS.confidence_set_full == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write(
                    "Input %s timed out while constructing confidence set. \n" % index
                )
            return
    AS.reduce()
    if AS.reduced == "T.O.":
        for f in [count_file, results_file]:
            with open(f, "a") as myfile:
                myfile.write("Input %s timed out while reducing set. \n" % index)
            return
    AS.compute_extreme_distributions()
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

    AS_tt = AS.time_taken
    headers += [len(AS.confidence_set_full), len(AS.reduced)]
    DRO_problem = DRMPNVP(
        T, W, w, p, h, b, PWL_gap, timeout - AS_tt, solver_cores, dist, AS
    )

    PWL_sol = DRO_problem.PWL_solve(AS.reduced)
    PWL_q, PWL_obj, PWL_worst, PWL_tt, PWL_TO = PWL_sol
    if PWL_q[0] != -1:
        PWL_obj = DRO_problem.cost_function(PWL_q, PWL_worst)
        objs = [DRO_problem.cost_function(PWL_q, o) for o in AS.confidence_set_full]
        PWL_true_worst = AS.confidence_set_full[np.argmax(objs)]
        PWL_true_worst_obj = np.max(objs)
        PWL_true_obj = np.round(DRO_problem.cost_function(PWL_q, omega_0), 5)
    else:
        PWL_obj = -1
        PWL_true_worst = PWL_worst
        PWL_true_worst_obj = -1
        PWL_true_obj = -1

    CS_sol = DRO_problem.CS_solve(AS.reduced[0])
    CS_q, CS_worst, CS_obj, CS_tt, CS_TO, CS_reason = CS_sol
    if CS_q[0] != -1:
        CS_obj = DRO_problem.cost_function(CS_q, CS_worst)
        objs = [DRO_problem.cost_function(CS_q, o) for o in AS.confidence_set_full]
        CS_true_worst = AS.confidence_set_full[np.argmax(objs)]
        CS_true_worst_obj = np.max(objs)
        CS_true_obj = np.round(DRO_problem.cost_function(CS_q, omega_0), 5)
    else:
        CS_obj = -1
        CS_true_worst = CS_worst
        CS_true_worst_obj = -1
        CS_true_obj = -1

    CS_reasons = ["TO", "repeat", "optimal", "k"]
    CS_reason = CS_reasons.index(CS_reason)

    fd_problem = MPNVP(
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
    fd_q = fd_problem.alfares_solve()
    e = time.perf_counter()
    fd_tt = np.round(e - s, 3)
    fd_obj = fd_problem.cost_function(fd_q, X.mle)

    objs = [fd_problem.cost_function(fd_q, o) for o in AS.confidence_set_full]
    fd_true_worst = AS.confidence_set_full[np.argmax(objs)]
    fd_true_worst_obj = np.max(objs)
    fd_true_obj = np.round(fd_problem.cost_function(fd_q, omega_0), 5)

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
        tuple(np.round(fd_q, 3)),
        tuple([tuple(i) for i in np.round(X.mle, 3)]),
        np.round(fd_obj, 5),
        tuple([tuple(i) for i in np.round(fd_true_worst, 3)]),
        np.round(fd_true_worst_obj, 5),
        fd_tt,
        fd_true_obj,
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


# T = int(sys.argv[1]) + 1
num_processors = 32
gurobi_cores = 4
loop_cores = int(num_processors / gurobi_cores)
timeout = 4 * 60 * 60

T_vals = range(2, 5)
# mu_0_range = range(3, 40)
# sig_0_range = range(3, 15)
mu_0_range = range(1, 21)
sig_0_range = range(1, 11)
num_omega0 = 3

PWL_gap_vals = list(reversed([0.1, 0.25, 0.5]))
gap = PWL_gap_vals[int(sys.argv[1]) - 1]
disc_pts_vals = [3, 5, 10]
p_range = list(100 * np.array(range(1, 3)))
h_range = list(100 * np.array(range(1, 3)))
b_range = list(100 * np.array(range(1, 3)))
W_range = [4000, 4000, 8000]
N_vals = [10, 25, 50]

omega0_all = [
    [
        (m, s)
        for (m, s) in mu_sig_combos(mu_0_range, sig_0_range, T_)
        if np.all(np.array([s[t] <= m[t] / 3 for t in range(T_)]))
    ]
    for T_ in T_vals
]

# omega0_all = [mu_sig_combos(mu_0_range, sig_0_range, T_) for T_ in T_vals]

omega0_vals = []
for T_ in T_vals:
    np.random.seed(T_vals.index(T_))
    indices = np.random.choice(range(len(omega0_all[T_vals.index(T_)])), num_omega0)
    omega0_vals.append([omega0_all[T_vals.index(T_)][i] for i in indices])

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
    "fd_q",
    "fd_worst",
    "fd_obj",
    "fd_true_worst",
    "fd_true_worst_obj",
    "fd_tt",
    "fd_true_obj",
    "MLE_neg",
]

num_reps = 1
num_instances = 10

if num_reps > 1:
    results_file = "results_reps.txt"
    count_file = "count_reps.txt"
    repeated_inputs = []
    counter = 0
    for T_ in T_vals:
        inps = [i for i in inputs if i[3] == T_][:num_instances]
        for rep in range(num_reps):
            repeated_inputs += [
                tuple([i[0], rep, num_reps] + list(i)[1:]) for i in inps
            ]
else:
    results_file = "results_normal_chi.txt"
    count_file = "count_normal_chi.txt"
    inps = inputs
    repeated_inputs = [tuple([i[0], 0, 1] + list(i)[1:]) for i in inps]

inputs = repeated_inputs

test_full = [
    i for i in inputs if (i[names.index("T")], i[names.index("n_pts")]) != (4, 10)
]

# test_full = inputs

continuing = True

if continuing:
    file1 = open(results_file, "r")
    lines = file1.readlines()
    file1.close()

    lines_new = []
    failed = []
    # names = eval(lines[0])
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
    # no_TO = df[(df.PWL_TO == 0) & (df.CS_TO == 0)]

    # done_twice = [(i, r) for (i, r) in list(zip(df.ind, df.rep))
    #             if df[(df.ind == i) & (df.rep == r)].shape[0] == 2]

    file1 = open(count_file, "r")
    lines = file1.readlines()
    file1.close()

    timed_out = []
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "timed" in line:
            i = [int(i) for i in line.split() if i.isdigit()][0]
            timed_out.append((i, 0))

    done = list(zip(df.ind, df.rep)) + failed + timed_out

    not_done = [i for i in test_full if (i[0], i[1]) not in done]

    test = not_done
else:
    test = test_full

# if T == 2 and continuing:
if gap == 0.1 and continuing:
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving the %s instances that didn't finish solving before. \n"
            % len(test)
        )

# test = [i for i in test if i[names.index("T")] == T]
test = [i for i in test if i[names.index("PWL_gap")] == gap]


# wipes results file
# if T == 2 and not continuing:
if gap == 0.1 and not continuing:
    open(count_file, "w").close()
    open(results_file, "w").close()
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving %s instances with the repeated sampling approach. \n"
            % len(test)
        )
    with open(results_file, "a") as myfile:
        myfile.write(str(names) + "\n")


if __name__ == "__main__":
    with Pool(processes=loop_cores) as p:
        res = list(p.imap(test_algorithms_mp, test))

# res = [test_algorithms(inp) for inp in test]
