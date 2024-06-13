import struct
import time

import numpy as np

from CHO import CHO_optimizor as optimizer

if __name__ == "__main__":
    MAX_ACCESS = 1000
    feasible = True
    iterations = 1
    optimizor_option = "CHO" # CHO or BHO

    C = np.load('../generateScenario/simulation_coverage_info.npy')
    LogStartingTime = time.time()
    co = optimizer(coverage_info=C, max_access=MAX_ACCESS, T=20)
    print('\033[91m' + f"Initializing optimizer {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')
    prev_d = co.optimize()
