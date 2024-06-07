import struct
import time

import numpy as np

from BHO import HO_optimizor as bhoOptimizer
from CHO_no_candidate import CHO_optimizor as choOptimizer


def split_C(C, iterations):
    C_list = []
    time_period = C.shape[2] // iterations
    cutting_list = []
    for i in range(iterations):
        cutting_list.append(i * time_period)
    cutting_list.append(C.shape[2])
    for i in range(iterations):
        C_list.append(C[:, :, max(cutting_list[i] - 1, 0):cutting_list[i + 1]])
    return C_list

if __name__ == "__main__":
    MAX_REQUESTS = 1000
    MAX_ACCESS = 1000
    feasible = True
    iterations = 1
    optimizor_option = "CHO" # CHO or BHO

    LogStartingTime = time.time()
    C = np.load('../generateScenario/optimizer_coverage_info.npy')
    C_list = split_C(C, iterations)
    print('\033[91m' + f"Load scenario costs {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')

    if optimizor_option == "BHO":
        # BHO
        for i in range(iterations):
            LogStartingTime = time.time()
            bo = bhoOptimizer(feasible=feasible, coverage_info=C, max_requests=MAX_REQUESTS)
            print('\033[91m' + f"Initializing optimizer {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')
            bo.optimize()

    elif optimizor_option == "CHO":
        # CHO
        prev_d = None
        for i in range(iterations):
            LogStartingTime = time.time()
            co = choOptimizer(feasible=feasible, coverage_info=C_list[i], max_requests=MAX_REQUESTS, max_access=MAX_ACCESS,
                              T304_slot=20)
            print('\033[91m' + f"Initializing optimizer {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')
            prev_d = co.optimize(i, prev_d)
