import numpy as np
import pandas as pd
import sys, os
from numba import njit
import itertools as it


def read_results(file, dist):
    file1 = open(file, "r")
    lines = file1.readlines()
    file1.close()
    if dist == "normal":
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
    elif dist == "poisson":
        names = [
            "ind",
            "rep",
            "num_reps",
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
            "lam0_obj",
            "lam0_q",
            "lam0_tt",
        ]
    lines_new = []
    for line in lines:
        line = line.rstrip("\n")
        if "failed" not in line:
            if type(eval(line)[0]) == str:
                names = eval(line)
            else: lines_new.append(eval(line))
    print("Dataset has %s rows." % len(lines_new))

    res_array = np.array(lines_new)
    df = pd.DataFrame(data=res_array, columns=names, index=res_array[:, 0])
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


@njit
def pois_conf_val(N, T, lam, lam_hat):
    val = 0
    for t in range(T):
        val += (N / lam_hat[t]) * (lam_hat[t] - lam[t]) ** 2
    return val


def mu_sig_combos(mu_0_range, sig_0_range, T):
    return [
        (mu_0, sig_0)
        for mu_0 in list(it.combinations_with_replacement(mu_0_range, T))
        for sig_0 in list(it.combinations_with_replacement(sig_0_range, T))
    ]
