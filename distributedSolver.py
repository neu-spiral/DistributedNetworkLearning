import cvxpy as cp
import pickle
import numpy as np
from ProbGenerate import Problem, Path
import logging, argparse
import time


class Gradient:

    def __init__(self, P):
        self.sourceRates = P.sourceRates
        self.sources = P.sources
        self.learners = P.learners
        self.bandwidth = P.bandwidth
        self.G = P.G
        self.paths = P.paths
        self.prior = P.prior
        self.T = P.T
        self.slToStp = P.slToStp
        self.types = P.types
        self.sourceParameters = P.sourceParameters

        self.Dependencies()

    def objG(self, features, n, noices, cov):
        temp = 0
        for s in self.sources:
            for i in range(n[s]):
                temp += np.dot(features[s][i], features[s][i].transpose()) / noices[s]
        temp = temp + np.linalg.inv(cov)
        temp = np.linalg.det(temp)
        obj = np.log(temp)
        return obj

    def objU(self, Y, N1, N2):
        obj = 0
        zeros = {}
        for s in self.sources:
            zeros[s] = 0
        for l in self.learners:
            for i in range(N1):
                n = self.generate_sample1(Y, l)
                for j in range(N2):
                    features = self.generate_sample2(n)
                    noices = self.prior['noice'][l]
                    cov = self.prior['cov'][l]
                    obj += self.objG(features, n, noices, cov) - self.objG(zeros, zeros, noices, cov)
        obj = obj / N1 / N2

        return obj

    def Dependencies(self):
        self.dependencies = {}
        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                path = p.path
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self.dependencies:
                        if (s, t) in self.dependencies[edge]:
                            self.dependencies[edge][(s, t)].append(p)
                        else:
                            self.dependencies[edge][(s, t)] = [p]
                    else:
                        self.dependencies[edge] = {(s, t): [p]}

    def getLearnerRate(self, Y, s, l):
        (s, t, p) = self.slToStp[(s, l)]
        return Y[(s, t)][p]

    def generate_sample1(self, Y, l):
        n = {}
        for s in self.sources:
            rate = self.getLearnerRate(Y, s, l)
            n[s] = np.random.poisson(rate * self.T)

        return n

    def generate_sample2(self, n):
        features = {}
        for s in self.sources:
            mean = self.sourceParameters['mean'][s]
            cov = self.sourceParameters['cov'][s]
            size = n[s]
            features[s] = np.random.multivariate_normal(mean, cov, size)

        return features

    def Estimate_Gradient(self, Y, head, N1, N2):

        Z = {}
        for (s, t) in self.paths:
            Z[(s, t)] = {}
            for p in self.paths[(s, t)]:
                Z[(s, t)][p] = 0

        for l in self.learners:
            noices = self.prior['noice'][l]
            cov = self.prior['cov'][l]

            for i in range(N1):
                n = self.generate_sample1(Y, l)
                n_h = {}  # each arrival is greater than head
                for s in n:
                    n_h[s] = max(head, n[s])
                for j in range(N2):
                    features = self.generate_sample2(n_h)
                    for s in self.sources:
                        n_copy = n.copy()
                        rate = self.getLearnerRate(Y, s, l)
                        delta = 0
                        for h in range(head):
                            n_copy[s] = h + 1
                            obj1 = self.objG(features, n_copy, noices, cov)
                            n_copy[s] = h
                            obj2 = self.objG(features, n_copy, noices, cov)

                            delta += (obj1 - obj2) * rate ** h * self.T ** (h + 1) / np.math.factorial(h) * \
                                     np.exp(-rate * self.T)
                        (s, t, p) = self.slToStp[(s, l)]
                        Z[(s, t)][p] += delta / N1 / N2

        return Z

    def adapt(self, Y, D, gamma):
        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                Y[(s, t)][p] += gamma * D[(s, t)][p]

    def plus(self, x, y):
        if x <= 0:
            return max(y, 0)
        else:
            return y

    def feasibility(self, Y):
        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                if not Y[(s, t)][p] >= 0:
                    logging.debug("Source {}, type {}, learner {} has negative value: {}".format(s, t, p.learner,
                                                                                                 Y[(s, t)][p]))

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = []
                for p in self.dependencies[e][(s, t)]:
                    temp_st.append(Y[(s, t)][p])
                temp += max(temp_st)
            if not temp <= self.bandwidth[e]:
                logging.debug("Edge {} with bandwidth {} has flow {}".format(e, self.bandwidth[e], temp))

        for (s, t) in self.paths:
            temp = 0
            for p in self.paths[(s, t)]:
                temp += Y[(s, t)][p]
            if not temp <= self.sourceRates[(s, t)]:
                logging.debug("Source {}, type {} with rate {} has flow {}".format(s, t, self.sourceRates[(s, t)],
                                                                                   temp))

    def alg(self, iterations, head, N1, N2, distributed):
        pass


class FrankWolf(Gradient):
    def find_max_distributed(self, Z, iterations):
        """
        Solve a linear programing given gradient Z: argmax_D D*Z
        Z: a dictionary

        return: a dictionary D
        """
        n = 10
        stepsize = 0.02

        Q_new, Q_old = {}, {}
        for e in self.dependencies:
            Q_new[e], Q_old[e] = 0, 0

        R_new, R_old = {}, {}
        for (s, t) in self.sourceRates:
            R_new[(s, t)], R_old[(s, t)] = 0, 0

        U_new, U_old = {}, {}
        D_new, D_old = {}, {}
        for (s, t) in self.paths:
            U_new[(s, t)], U_old[(s, t)] = {}, {}
            D_new[(s, t)], D_old[(s, t)] = {}, {}
            for p in self.paths[(s, t)]:
                U_new[(s, t)][p], U_old[(s, t)][p] = 0, 0
                D_new[(s, t)][p], D_old[(s, t)][p] = 0, 0

        for i in range(iterations):
            temp_Q = {}
            for e in self.dependencies:
                Q_new[e] = Q_old[e]
                temp = 0
                for (s, t) in self.dependencies[e]:
                    temp_st = 0
                    for p in self.dependencies[e][(s, t)]:
                        temp_st += pow(D_old[(s, t)][p], n)
                    temp += pow(temp_st, 1. / n)
                temp -= self.bandwidth[e]
                temp_Q[e] = temp
                temp = np.expm1(temp)
                Q_new[e] += stepsize / (i+1) * self.plus(Q_old[e], temp)

            temp_R = {}
            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in self.paths[(s, t)]:
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                temp_R[(s, t)] = temp
                temp = np.expm1(temp)
                R_new[(s, t)] += stepsize / (i+1) * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in self.paths[(s, t)]:
                    U_new[(s, t)][p] = U_old[(s, t)][p] + stepsize / (i+1) * self.plus(U_old[(s, t)][p],
                                                                               np.expm1(-D_old[(s, t)][p]))

                    temp = 0
                    path = p.path
                    for j in range(len(path) - 1):
                        edge = (path[j], path[j + 1])
                        temp_e = 0
                        for p_e in self.dependencies[edge][(s, t)]:
                            temp_e += pow(D_old[(s, t)][p_e], n)
                        if temp_e:
                            temp += Q_old[edge] * np.exp(temp_Q[edge]) * pow(temp_e, 1. / n - 1)
                        else:
                            temp += Q_old[edge] * np.exp(temp_Q[edge])
                    temp *= pow(D_old[(s, t)][p], n - 1)

                    D_new[(s, t)][p] = D_old[(s, t)][p] + stepsize / (i+1) * (
                                Z[(s, t)][p] - temp - R_old[(s, t)] * np.exp(temp_R[(s, t)])
                                + U_old[(s, t)][p]) * np.exp(-D_old[(s, t)][p])

                U_old[(s, t)] = U_new[(s, t)].copy()
                D_old[(s, t)] = D_new[(s, t)].copy()
            Q_old, R_old = Q_new.copy(), R_new.copy()

        return D_new

    def find_max(self, Z):
        constr = []

        D = {}
        for (s, t) in self.paths:
            D[(s, t)] = {}
            for p in self.paths[(s, t)]:
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                st_size = len(self.dependencies[e][(s, t)])
                temp_st = cp.Variable(st_size)
                for i in range(st_size):
                    p = self.dependencies[e][(s, t)][i]
                    constr.append(temp_st[i] == D[(s, t)][p])
                temp += cp.max(temp_st)
            constr.append(temp <= self.bandwidth[e])

        for (s, t) in self.paths:
            temp = 0
            for p in self.paths[(s, t)]:
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                obj += D[(s, t)][p] * Z[(s, t)][p]

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        print("status: ", problem.status)

        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                D[(s, t)][p] = D[(s, t)][p].value

        return D

    def alg(self, iterations, head, N1, N2, distributed):

        Y = {}
        for (s, t) in self.paths:
            Y[(s, t)] = {}
            for p in self.paths[(s, t)]:
                Y[(s, t)][p] = 0

        gamma = 1. / iterations
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            if distributed:
                D = self.find_max_distributed(Z, iterations)
            else:
                D = self.find_max(Z)
            self.adapt(Y, D, gamma)
            # print(t, Y)

        return Y


class ProjectAscent(Gradient):

    def project_distributed(self, Y, iterations):
        """
        Solve a project given Y: argmin_D (D-Y)^2
        Y: a dictionary
        return: a dictionary D
        """
        n = 10
        stepsize = 0.01

        Q_new, Q_old = {}, {}
        for e in self.dependencies:
            Q_new[e], Q_old[e] = 0, 0

        R_new, R_old = {}, {}
        for (s, t) in self.sourceRates:
            R_new[(s, t)], R_old[(s, t)] = 0, 0

        U_new, U_old = {}, {}
        D_new, D_old = {}, {}
        for (s, t) in self.paths:
            U_new[(s, t)], U_old[(s, t)] = {}, {}
            D_new[(s, t)], D_old[(s, t)] = {}, {}
            for p in self.paths[(s, t)]:
                U_new[(s, t)][p], U_old[(s, t)][p] = 0, 0
                D_new[(s, t)][p], D_old[(s, t)][p] = 0, 0

        for i in range(1000):
            for e in self.dependencies:
                Q_new[e] = Q_old[e]
                temp = 0
                for (s, t) in self.dependencies[e]:
                    temp_st = 0
                    for p in self.dependencies[e][(s, t)]:
                        temp_st += pow(D_old[(s, t)][p], n)
                    temp += pow(temp_st, 1. / n)
                temp -= self.bandwidth[e]
                Q_new[e] += 100 * stepsize * self.plus(Q_old[e], temp)

            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in self.paths[(s, t)]:
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                R_new[(s, t)] += 100 * stepsize * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in self.paths[(s, t)]:
                    U_new[(s, t)][p] = U_old[(s, t)][p] + 100 * stepsize * self.plus(U_old[(s, t)][p],
                                                                                            -D_old[(s, t)][p])

                    temp = 0
                    path = p.path
                    for j in range(len(path) - 1):
                        edge = (path[j], path[j + 1])
                        temp_e = 0
                        for p_e in self.dependencies[edge][(s, t)]:
                            temp_e += pow(D_old[(s, t)][p_e], n)
                        if temp_e:
                            temp += Q_old[edge] * pow(temp_e, 1. / n - 1)
                        else:
                            temp += Q_old[edge]
                    temp *= pow(D_old[(s, t)][p], n - 1)

                    D_new[(s, t)][p] = D_old[(s, t)][p] + stepsize * (
                                -2 * (D_old[(s, t)][p] - Y[(s, t)][p]) - temp -
                                R_old[(s, t)] + U_old[(s, t)][p])

                U_old[(s, t)] = U_new[(s, t)].copy()
                D_old[(s, t)] = D_new[(s, t)].copy()
            Q_old, R_old = Q_new.copy(), R_new.copy()

        return D_new

    def project(self, Y):
        constr = []

        D = {}
        for (s, t) in self.paths:
            D[(s, t)] = {}
            for p in self.paths[(s, t)]:
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                st_size = len(self.dependencies[e][(s, t)])
                temp_st = cp.Variable(st_size)
                for i in range(st_size):
                    p = self.dependencies[e][(s, t)][i]
                    constr.append(temp_st[i] == D[(s, t)][p])
                temp += cp.max(temp_st)
            constr.append(temp <= self.bandwidth[e])

        for (s, t) in self.paths:
            temp = 0
            for p in self.paths[(s, t)]:
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                obj += (D[(s,t)][p] - Y[(s,t)][p]) ** 2

        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve()
        print("status: ", problem.status)

        for (s, t) in self.paths:
            for p in self.paths[(s, t)]:
                D[(s, t)][p] = D[(s, t)][p].value

        return D

    def alg(self, iterations, head, N1, N2, distributed):
        Y = {}
        for (s, t) in self.paths:
            Y[(s, t)] = {}
            for p in self.paths[(s, t)]:
                Y[(s, t)][p] = 0

        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            self.adapt(Y, Z, 1. / (t + 1))
            if distributed:
                Y = self.project_distributed(Y, iterations)
            else:
                Y = self.project(Y)
            self.feasibility(Y)
            # print(t, Y)

        return Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--debug_level', default='DEBUG', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    fname = 'Problem/Problem_' + args.graph_type
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)

    alg1 = FrankWolf(P)
    t1 = time.time()
    Y1 = alg1.alg(iterations=50, head=20, N1=20, N2=20, distributed=True)
    obj1 = alg1.objU(Y=Y1, N1=10, N2=10)
    t2 = time.time()
    print(t2-t1, Y1, obj1)
    Y2 = alg1.alg(iterations=50, head=20, N1=20, N2=20, distributed=False)
    obj2 = alg1.objU(Y=Y2, N1=20, N2=20)
    t3 = time.time()
    print(t3-t2, Y2, obj2)

    alg4 = ProjectAscent(P)
    t4 = time.time()
    Y6 = alg4.alg(iterations=50, head=20, N1=20, N2=20, distributed=True)
    obj6 = alg4.objU(Y=Y6, N1=20, N2=20)
    t5 = time.time()
    print(t5-t4, Y6, obj6)
    Y7 = alg4.alg(iterations=50, head=20, N1=20, N2=20, distributed=False)
    obj7 = alg4.objU(Y=Y7, N1=20, N2=20)
    t6 = time.time()
    print(t6-t5, Y7, obj7)

    fname = 'Result/Result_' + args.graph_type
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump([(Y1, obj1), (Y2, obj2), (Y6, obj6), (Y7, obj7)], f)
