import numpy as np
import pandas as pd
import sys, os
from numba import njit
import itertools as it


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
