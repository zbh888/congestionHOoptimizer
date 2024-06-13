import copy
import pickle
import time

import gurobipy
import numpy as np
from gurobipy import GRB, quicksum


class CHO_optimizor:
    # ========= config ============
    def generate_L(self, C):
        N_UE, N_SAT, N_TIME = C.shape
        L = np.zeros((N_UE, N_SAT, N_TIME), dtype=int)
        for u in range(N_UE):
            for s in range(N_SAT):
                for t in range(N_TIME - 1):
                    if C[u, s, t] == 1 and C[u, s, t + 1] == 0:
                        L[u, s, t] = 1
        return L

    def generate_important_time(self, LL):
        L = np.array(LL)
        V = []
        original2important = np.full((L.shape[0], L.shape[2]), -1)
        for u_i, U in enumerate(L):
            V_U = []
            count = 0
            access_period = 0
            for t_i, T in enumerate(U.T):
                # TODO  t_i == len(U.T)-1 this might not needed
                ho_time = np.sum(T)
                if ho_time > 0 or t_i == 0 or t_i == len(U.T) - 1 or access_period > 0:
                    original2important[u_i, t_i] = count
                    count += 1
                    V_U.append(T)
                    access_period -= 1
                    if ho_time > 0:
                        access_period = self.T
            V_U_T = np.array(V_U).T
            V.append(V_U_T.tolist())
        return V, original2important

    def important2original(self, o2i):
        i2o = []
        for U in o2i:
            T = []
            for t_i, t in enumerate(U):
                if t > -1:
                    T.append(t_i)
            i2o.append(T)
        return i2o

    def generate_important_coverage(self, CC, o2i):
        C = []
        for u_i, U in enumerate(CC):
            U_C = []
            for t_i, T in enumerate(U.T):
                if o2i[u_i][t_i] > -1:
                    U_C.append(T)
            C.append((np.array(U_C).T).tolist())
        return C

    def generate_UE_N_TIME(self, ot):
        l = []
        for UE in range(self.N_UE):
            if self.o2i[UE][ot] > -1:
                l.append((UE, self.o2i[UE][ot]))
        return l

    def save_one_variable(self, var):
        array = []
        # Assign values from self.C[u][s] to the corresponding positions in the array
        for u in range(self.N_UE):
            array_u = []
            for s in range(self.N_SAT):
                array_s = [0] * len(self.C[u][s])
                for t in range(len(self.C[u][s])):
                    array_s[t] = var[u, s, t].x
                array_u.append(array_s)
            array.append(array_u)
        return array

    # ========= optimize ============
    def __init__(self, feasible, coverage_info, max_requests, max_access, T):
        self.T = T
        self.original_C = copy.deepcopy(coverage_info)
        self.N_UE, self.N_SAT, self.N_TIME = self.original_C.shape
        LL = self.generate_L(self.original_C)  # takes a lot of time
        self.L, self.o2i = self.generate_important_time(LL)  # takes a lot of time
        self.C = self.generate_important_coverage(self.original_C, self.o2i)
        self.i2o = self.important2original(self.o2i)

        self.MAX_REQUESTS = max_requests
        self.MAX_ACCESS = max_access
        self.feasible = feasible

    def optimize(self, iteration, prev_d):
        N_candidate = 3
        LogStartingTime = time.time()
        UE_SAT_TIME = list(set([(u, s, t) for u in list(range(self.N_UE)) for s in list(range(self.N_SAT)) for t in
                                list(range(len(self.C[u][s])))]))
        UE_SAT = list(set([(u, s) for u in list(range(self.N_UE)) for s in list(range(self.N_SAT))]))
        UE_TIME = list(set([(u, t) for u in list(range(self.N_UE)) for s in list(range(self.N_SAT)) for t in
                            list(range(len(self.C[u][s])))]))

        # variables
        mdl = gurobipy.Model("CHO")
        o = mdl.addVar(vtype=GRB.CONTINUOUS, name="objective")
        m = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="m")
        d = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="h")
        h = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="h")
        hh = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="hh")
        y = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="y")
        yy = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="yy")
        x = mdl.addVars(UE_SAT_TIME, vtype=GRB.CONTINUOUS, name="xx")
        xx = mdl.addVars(UE_SAT_TIME, vtype=GRB.BINARY, name="xx")
        z = mdl.addVars(UE_SAT_TIME, vtype=GRB.CONTINUOUS, name="z")

        HT = mdl.addVars(UE_SAT, vtype=GRB.CONTINUOUS, name="a")
        HHT = mdl.addVars(UE_SAT, vtype=GRB.CONTINUOUS, name="a")
        XT = mdl.addVars(UE_SAT, vtype=GRB.CONTINUOUS, name="a")
        XXT = mdl.addVars(UE_SAT, vtype=GRB.CONTINUOUS, name="a")
        # if self.feasible:
        #     f = mdl.addVars(list(range(self.N_UE)), vtype=GRB.CONTINUOUS, name="f")
        #     M = 1e6

        mdl.update()

        # objective
        # mdl.modelSense = GRB.MAXIMIZE
        # if self.feasible:
        #     mdl.setObjective(quicksum((AccessT[u, s] - hoT[u, s]) for u, s in UE_SAT) + M * f.sum())
        # else:
        #     mdl.setObjective(quicksum((AccessT[u, s] - hoT[u, s]) for u, s in UE_SAT))

        # # constraints
        # if self.feasible:
        #     for u in list(range(self.N_UE)):
        #         max_u_time_dummy = len(self.C[u][self.N_SAT - 1])
        #         mdl.addConstr(f[u] == h[u, self.N_SAT - 1, max_u_time_dummy - 1], name="feasible")

        # if iteration != 0:
        #     d_h = prev_d['h']
        #     for u, s in UE_SAT:
        #         mdl.addConstr(((h[u, s, 0] == int(d_h[u][s][-1]))), name="iteration")
        for u,s,t in UE_SAT_TIME:
            mdl.addConstr((x[u, s, t] == m[u, s, t] * self.L[u][s][t]), name="1")
            mdl.addConstr((z[u, s, t] == (2+N_candidate)*x[u, s, t] + 
                           y[u, s, t] +
                           h[u, s, t] + 
                           2* xx[u, s, t] + 
                           yy[u, s, t] + 
                           3* hh[u, s, t]), name="1")
            

        for u, t in UE_TIME:
            mdl.addConstr(quicksum(h[u, s, t] for s in range(self.N_SAT)) <= 1, name="2")
            mdl.addConstr(quicksum(m[u, s, t] for s in range(self.N_SAT)) == 1, name="3")
            mdl.addConstr((quicksum(h[u, s, t] for s in range(self.N_SAT))
                        == quicksum(x[u, s, t] for s in range(self.N_SAT))), name="4")
            mdl.addConstr((quicksum(y[u, s, t] for s in range(self.N_SAT))
                        == (N_candidate-1)*quicksum(h[u, s, t] for s in range(self.N_SAT))), name="4")
            mdl.addConstr((quicksum(yy[u, s, t] for s in range(self.N_SAT))
                        == (N_candidate-1)*quicksum(hh[u, s, t] for s in range(self.N_SAT))), name="4")
            mdl.addConstr((quicksum(xx[u, s, t] for s in range(self.N_SAT))
                        == quicksum(hh[u, s, t] for s in range(self.N_SAT))), name="4")

        for u, s, t in UE_SAT_TIME:
            mdl.addConstr(((h[u, s, t] <= self.C[u][s][t])), name="BC11")
            mdl.addConstr(((m[u, s, t] <= self.C[u][s][t])), name="BC11")
            mdl.addConstr(((hh[u, s, t] <= self.C[u][s][t])), name="BC11")
            mdl.addConstr(((hh[u, s, t] + h[u, s, t] <= 1)), name="BC7")
            mdl.addConstr(((h[u, s, t] + y[u, s, t] <= 1)), name="BC7")
            mdl.addConstr(((x[u, s, t] + y[u, s, t] <= 1)), name="BC7")
            mdl.addConstr(((y[u, s, t] <= 1 - d[u,s,t])), name="BC7")
            mdl.addConstr(((yy[u, s, t] <= d[u,s,t])), name="BC7")
            if t != 0:
                mdl.addConstr((m[u, s, t] == quicksum(h[u, s, t - 1] for s in range(self.N_SAT)) * h[u, s, t - 1]
                               + (1 - quicksum(h[u, s, t - 1] for s in range(self.N_SAT))) * m[u, s, t - 1]),
                              name="AC4")
                mdl.addConstr((d[u, s, t] == (1 - yy[u,s,t-1])*(d[u,s,t-1]+y[u,s,t-1])), name="BC7")

        for u, s in UE_SAT:
            mdl.addConstr(HT[u, s] == quicksum((self.i2o[u][nt] * h[u, s, nt]) for nt in range(len(self.C[u][s]))),
                          name="BC9")
            mdl.addConstr(
                HHT[u, s] == quicksum(((self.i2o[u][nt] * hh[u, s, nt]) for nt in range(len(self.C[u][s])))),
                name="BC9")
            mdl.addConstr(XT[u, s] == quicksum((self.i2o[u][nt] * x[u, s, nt]) for nt in range(len(self.C[u][s]))),
                          name="BC9")
            mdl.addConstr(
                XXT[u, s] == quicksum(((self.i2o[u][nt] * xx[u, s, nt]) for nt in range(len(self.C[u][s])))),
                name="BC9")
            mdl.addConstr(HHT[u, s] - HT[u, s] <= self.T, name="BC10")
            mdl.addConstr(HHT[u, s] - HT[u, s] >= 0, name="BC10")
            mdl.addConstr(XXT[u, s] - XT[u, s] <= self.T, name="BC10")
            mdl.addConstr(XXT[u, s] - XT[u, s] >= 0, name="BC10")
            mdl.addConstr(quicksum(h[u, s, nt] for nt in range(len(self.C[u][s])))
                          == quicksum(hh[u, s, nt] for nt in range(len(self.C[u][s]))), name="BC6")
            mdl.addConstr(quicksum(y[u, s, nt] for nt in range(len(self.C[u][s])))
                          == quicksum(yy[u, s, nt] for nt in range(len(self.C[u][s]))), name="BC6")
            mdl.addConstr(quicksum(xx[u, s, nt] for nt in range(len(self.C[u][s])))
                          == quicksum(x[u, s, nt] for nt in range(len(self.C[u][s]))), name="BC6")

        for ot in range(self.N_TIME):
            UE_N_TIME = self.generate_UE_N_TIME(ot)
            Number_of_satellite = self.N_SAT
            if self.feasible:
                Number_of_satellite = self.N_SAT - 1
            for s in range(Number_of_satellite):
                if len(UE_N_TIME) >= 1:
                    mdl.addConstr((quicksum((hh[u, s, t]) for u, t in UE_N_TIME)
                                   <= self.MAX_ACCESS), name="BC8")

        # Maximum requests of $s$ allowed at $t$
        # for ot in range(self.N_TIME):
        #     UE_N_TIME = self.generate_UE_N_TIME(ot)
        #     Number_of_satellite = self.N_SAT
        #     if self.feasible:
        #         Number_of_satellite = self.N_SAT - 1
        #     for s in range(Number_of_satellite):
        #         if len(UE_N_TIME) >= 1:
        #             mdl.addConstr((quicksum((h[u, s, t] * self.L[u][s][t] + 2*x[u, s, t]) for u, t in UE_N_TIME)
        #                            <= self.MAX_REQUESTS), name="AC5")
        #             mdl.addConstr(z[s, ot] == quicksum(x[u, s, t] for u, t in UE_N_TIME), name="objective")
        #             objective_var.append(z[s,ot])

        mdl.addConstr(o == gurobipy.max_(z), "max_constraint")
        mdl.setObjective(o, sense=GRB.MINIMIZE)
        print('\033[91m' + f"Adding constraints costs {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')

        LogStartingTime = time.time()
        mdl.optimize()
        print('\033[91m' + f"Optimization costs {round(time.time() - LogStartingTime, 1)} seconds" + '\033[0m')

        self.mdl = mdl
        # print(z)
        # self.z = z
        # self.m = m
        # self.a = a
        dict = {}
        dict["m"] = self.save_one_variable(m)
        # dict["h"] = self.save_one_variable(h)
        # dict["a"] = self.save_one_variable(a)
        # # if self.feasible:
        # #     dict["f"] = self.save_one_variable(f)
        # dict["MAX_REQUESTS"] = self.MAX_REQUESTS
        # dict["MAX_ACCESS"] = self.MAX_ACCESS
        # dict["C"] = self.C
        # dict["o2i"] = self.o2i
        # dict["i2o"] = self.i2o
        # dict["N_TIME"] = self.N_TIME
        # dict["L"] = self.L
        # with open('cho' + str(iteration) + '.pkl', 'wb') as outp:
        #     pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
        return dict

    def show_redundant_const(self):
        print("#####Redundant Constraints#####")
        for constr in self.mdl.getConstrs():
            if constr.Slack < 1e-6:
                print(f"Constraint: {constr.ConstrName}")
        print()

