import struct
import time

import numpy as np

from CHO import CHO_optimizor as optimizer

if __name__ == "__main__":
    MAX_ACCESS = 2
    optimizor_option = "CHO" # CHO or BHO
    T = 20
    C = np.load('../generateScenario/simulation_coverage_info.npy')
    last_elements_repeated = np.repeat(C[:, :, -1:], 2*T, axis=2)
    C = np.concatenate((C, last_elements_repeated), axis=2)
    LogStartingTime = time.time()
    co = optimizer(coverage_info=C, max_access=MAX_ACCESS, T=T)
    print('\033[91m' + f"Initializing optimizer {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')
    prev_d = co.optimize()
