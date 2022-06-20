from multiprocessing import Pool
import sys, os
from src.ambiguity_set import *
from src.demand_RV import *
from src.DRMPNVP import *
from src.MPNVP import *
from src.functions import *
from scipy.optimize import minimize

def test_algorithms(inp):
    ind, dist, p, h, b, w, T, W, mu, sig, budget_tol, tau = inp
    headers= [ind, p, h, b, tuple(np.round(w, 3)), T, W, mu, tuple(np.round(sig, 3)), budget_tol, tau]

    if dist == "normal":
        omega = (mu, sig)
    if dist == "poisson":
        omega = mu

    PWL_gap = 0.1
    timeout = 14400
    solver_cores = 1
    X = demand_RV(dist, T, omega, N=1, seed=ind)

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
        omega,
        budget_tol,
        tau,
        alpha_min=1e-100,
        alpha_init=0.1,
    )

    s=time.perf_counter()
    alfares_q = fd_problem.alfares_solve()
    e=time.perf_counter()
    t_alf = np.round(e-s, 3)
    if type(alfares_q) is not str:
        alfares_obj = np.round(fd_problem.cost_function(alfares_q, omega), 5)
        alfares_budget = sum(w * alfares_q)
        alfares_budget_gap = "{:.5e}".format(alfares_budget - W)

    s=time.perf_counter()
    res = minimize(fd_problem.cost_function, x0=T*[0], args=(omega, ), 
        bounds = T * [(0, None)],
        method="trust-constr",
        options={'gtol': budget_tol},
        constraints=({'type': 'ineq', 'fun': lambda x:  W - sum(w * x)})
    )
    TC_q = res.x
    e=time.perf_counter()
    TC_obj = np.round(fd_problem.cost_function(TC_q, omega), 5)
    TC_budget = sum(w * TC_q)
    TC_budget_gap = "{:.5e}".format(TC_budget - W)
    t_TC = np.round(e-s, 3)
    
    s=time.perf_counter()
    res = minimize(fd_problem.cost_function, x0=T*[0], args=(omega, ), 
        bounds = T * [(0, None)],
        method="SLSQP",
        constraints=({'type': 'ineq', 'fun': lambda x:  W - sum(w * x)})
    )
    SLSQP_q = res.x
    e=time.perf_counter()
    SLSQP_obj = np.round(fd_problem.cost_function(TC_q, omega), 5)
    SLSQP_budget = sum(w * SLSQP_q)
    SLSQP_budget_gap = "{:.5e}".format(SLSQP_budget - W)
    t_SLSQP = np.round(e-s, 3)

    DRO_problem = DRMPNVP(
        T, W, w, p, h, b, PWL_gap, timeout, solver_cores, dist, AS=[]
    )

    # PWL solution for MLE distribution
    PWL_sol = DRO_problem.PWL_solve([omega])
    PWL_q, PWL_obj, PWL_worst, PWL_tt, PWL_TO = PWL_sol
    PWL_obj = np.round(fd_problem.cost_function(PWL_q, omega), 5)
    PWL_budget = sum(w * PWL_q)
    PWL_budget_gap = "{:.5e}".format(PWL_budget - W)
    t_PWL = np.round(PWL_tt, 3)
    
    with open("fast_alfares_count.txt", "a") as myfile:
        myfile.write("Finished input %s. \n" %ind)
    
    res_list = headers + [tuple(np.round(alfares_q, 5)), alfares_obj,
        alfares_budget_gap, t_alf,
        tuple(np.round(TC_q, 5)), TC_obj,
        TC_budget_gap, t_TC,
        tuple(np.round(SLSQP_q, 5)), SLSQP_obj,
        SLSQP_budget_gap, t_SLSQP,
        tuple(np.round(PWL_q, 5)), PWL_obj,
        PWL_budget_gap, t_PWL
        ]

    with open("fast_alfares_results.txt", "a") as res_file:
        res_file.write(str(res_list) + "\n")

    return res_list

    
import logging
def test_algorithms_mp(inp):
    ind = inp[0]
    try:
        return test_algorithms(inp)
    except Exception:
        with open("fast_alfares_count.txt", "a") as res_file:
            res_file.write("Input %s failed.\n" %ind)
        logging.exception("Input %s failed" % (ind))
        
        
cores = 32
p_range = range(1, 4)
h_range = range(1, 4)
b_range = range(1, 4)
T_vals = range(3, 7)
w_vals = ([list(2 * np.array(list(reversed(range(1, T + 1))))) for T in T_vals] +
          [[1/t for t in range(1,T+1)] for T in T_vals] 
         )
W_vals = [5, 10, 25, 50]
mu_vals = []
for T in T_vals:
    all_vals = (list(it.combinations_with_replacement([5, 10, 20], r=T)) 
        +  list(it.combinations_with_replacement([5, 10, 20], r=T)))
    for mu in all_vals:
        if mu not in mu_vals:
            mu_vals.append(mu)
        
sig_vals = [tuple(np.array(mu) / 4) for mu in mu_vals]
budget_tols = [0, 1e-10]
tau_vals = [0.75, 0.5, 0.25]
inputs = [
    (dist, p, h, b, w, T, W, mu, tuple(np.array(mu) / 4), budget_tol, tau)
    for dist in ["normal", "poisson"]
    for p in p_range for h in h_range for b in b_range
    for T in T_vals for w in w_vals 
    for W in W_vals for mu in mu_vals
    for budget_tol in budget_tols
    for tau in tau_vals
    if len(w) == T and len(mu) == T
    and b > max([w[t] - w[t+1] for t in range(T-1)])
]


for i in range(len(inputs)):
    inputs[i] = tuple([i] + list(inputs[i]))
num_inputs = len(inputs)

names = ["ind", "p", "h", "b", "w", "T", "W", "mu", "sig", "budget_tol", "tau",
         "q_alf", "alf_obj", "alf_gap", "t_alf", "TC_q", "TC_obj", "TC_budget_gap", "t_TC",
        "SLSQP_q", "SLSQP_obj", "SLSQP_budget_gap", "t_SLSQP",
        "PWL_q", "PWL_obj", "PWL_budget_gap", "t_PWL"
        ]

test = inputs

continuing = True

if continuing:
    df = read_results("fast_alfares_results.txt")
    done = [i for i in test if i[0] not in list(df.ind)]
    with open("fast_alfares_count.txt", "a") as myfile:
        myfile.write("About to start solving the %s instances.\
        that did not solve before.\n" %len(test))
else:
    open('fast_alfares_count.txt', 'w').close()
    open('fast_alfares_results.txt', 'w').close()
    with open("fast_alfares_count.txt", "a") as myfile:
        myfile.write("About to start solving %s  newsvendor instances. \n" %len(inputs))
    with open("fast_alfares_results.txt", "a") as myfile:
        myfile.write(str(names) + "\n")
    
if __name__ ==  '__main__': 
    with Pool(processes=cores) as p:
        res = list(p.imap(test_algorithms_mp, test))
