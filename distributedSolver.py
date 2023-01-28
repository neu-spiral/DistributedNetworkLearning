import cvxpy as cp
import pickle
import numpy as np
from ProbGenerate import Problem, Path
import logging, argparse
import time


class Learning:
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
        self.sourceParameters = P.sourceParameters

        self.Dependencies()

    def Dependencies(self):
        """ For each edge, dependencies stores which (s,t),p would be affected. """
        self.dependencies = {}
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                path = self.paths[(s, t)][p].path
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
            noices = self.prior['noice'][l]
            cov = self.prior['cov'][l]
            obj2 = self.objG(zeros, zeros, noices, cov)
            for i in range(N1):
                n = self.generate_sample1(Y, l)
                for j in range(N2):
                    features = self.generate_sample2(n)
                    obj1 = self.objG(features, n, noices, cov)
                    obj += obj1 - obj2
        obj = obj / N1 / N2

        return obj

    def generate_sample1(self, Y, l):
        n = {}
        for s in self.sources:
            rate = self.getLearnerRate(Y, s, l)
            if rate < 0:
                rate = 0
            n[s] = np.random.poisson(rate * self.T)

        return n

    def generate_sample2(self, n):
        features = {}
        for s in self.sources:
            mean = self.sourceParameters['mean'][s]
            cov = self.sourceParameters['cov'][s]
            size = n[s]
            features[s] = np.random.multivariate_normal(mean, cov, size)
            features[s] = features[s].reshape((n[s], len(mean), 1))

        return features

    def plus(self, x, y):
        if x <= 0:
            return max(y, 0)
        else:
            return y

    def feasibility(self, Y, tol=1e-3):
        feasibility = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                if Y[(s, t)][p] < -tol:
                    logging.debug("Source {}, type {}, learner {} has negative value: {}".format(s, t,
                                                                                                 self.paths[(s, t)][
                                                                                                     p].learner,
                                                                                                 Y[(s, t)][p]))
                    feasibility += - Y[(s, t)][p]

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = []
                for p in self.dependencies[e][(s, t)]:
                    temp_st.append(Y[(s, t)][p])
                temp += max(temp_st)
            if temp - self.bandwidth[e] > tol:
                logging.debug("Edge {} with bandwidth {} has flow {}".format(e, self.bandwidth[e], temp))
                feasibility += temp - self.bandwidth[e]

        for (s, t) in self.paths:
            temp = 0
            for p in range(len(self.paths[(s, t)])):
                temp += Y[(s, t)][p]
            if temp - self.sourceRates[(s, t)] > tol:
                logging.debug("Source {}, type {} with rate {} has flow {}".format(s, t, self.sourceRates[(s, t)],
                                                                                   temp))
                feasibility += temp - self.sourceRates[(s, t)]

        return feasibility


class Gradient(Learning):
    def Estimate_Gradient(self, Y, head, N1, N2):

        Z = {}
        for (s, t) in self.paths:
            Z[(s, t)] = {}
            for p in range(len(self.paths[(s, t)])):
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
            for p in range(len(self.paths[(s, t)])):
                Y[(s, t)][p] += gamma * D[(s, t)][p]

    def alg(self, iterations, head, N1, N2, stepsize):
        pass

    def Lagrangian(self, Z, D, Q, R, U, n):
        pass


class FrankWolf(Gradient):
    def Lagrangian(self, Z, D, Q, R, U, n):
        L = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L += D[(s, t)][p] * Z[(s, t)][p]

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = 0
                for p in self.dependencies[e][(s, t)]:
                    temp_st += pow(D[(s, t)][p], n)
                temp += pow(temp_st, 1. / n)
            temp -= self.bandwidth[e]
            temp = np.expm1(temp)
            L -= Q[e] * temp

        for (s, t) in self.paths:
            temp = 0
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            temp -= self.sourceRates[(s, t)]
            temp = np.expm1(temp)
            L -= R[(s, t)] * temp

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L -= U[(s, t)][p] * np.expm1(-D[(s, t)][p])
        return L

    def find_max_distributed(self, Z, iterations, stepsize):
        """
        Solve a linear programing given gradient Z: argmax_D D*Z
        Z: a dictionary

        return: a dictionary D
        """
        n = 10

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
            for p in range(len(self.paths[(s, t)])):
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
                Q_new[e] += stepsize * self.plus(Q_old[e], temp)

            temp_R = {}
            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in range(len(self.paths[(s, t)])):
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                temp_R[(s, t)] = temp
                temp = np.expm1(temp)
                R_new[(s, t)] += stepsize * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in range(len(self.paths[(s, t)])):
                    U_new[(s, t)][p] = U_old[(s, t)][p] + stepsize * self.plus(U_old[(s, t)][p],
                                                                               np.expm1(-D_old[(s, t)][p]))

                    temp = 0
                    path = self.paths[(s, t)][p].path
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

                    D_new[(s, t)][p] = D_old[(s, t)][p] + stepsize * (
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
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            '''Realize it using max function instead of approximation'''
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
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                obj += D[(s, t)][p] * Z[(s, t)][p]

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        logging.debug("status: " + problem.status)
        logging.debug("optimal values: " + str(problem.value))

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = D[(s, t)][p].value

        return D

    def alg(self, iterations, head, N1, N2, stepsize):

        Y = {}
        for (s, t) in self.paths:
            Y[(s, t)] = {}
            for p in range(len(self.paths[(s, t)])):
                Y[(s, t)][p] = 0

        gamma = 1. / iterations
        feasibilities = []
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            if stepsize:
                D = self.find_max_distributed(Z, 1000, stepsize)
                feasibility = self.feasibility(D)
                feasibilities.append(feasibility)
            else:
                D = self.find_max(Z)
            self.adapt(Y, D, gamma)
            logging.debug(t)

        feasibility = self.feasibility(Y)
        feasibilities.append(feasibility)
        return Y, feasibilities


class ProjectAscent(Gradient):
    def Lagrangian(self, Z, D, Q, R, U, n):
        L = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L -= (D[(s, t)][p] - Z[(s, t)][p]) ** 2

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = 0
                for p in self.dependencies[e][(s, t)]:
                    temp_st += pow(D[(s, t)][p], n)
                temp += pow(temp_st, 1. / n)
            temp -= self.bandwidth[e]
            L -= Q[e] * temp

        for (s, t) in self.paths:
            temp = 0
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            temp -= self.sourceRates[(s, t)]
            L -= R[(s, t)] * temp

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L += U[(s, t)][p] * U[(s, t)][p]
        return L

    def project_distributed(self, Y, iterations, stepsize):
        """
        Solve a project given Y: argmin_D (D-Y)^2
        Y: a dictionary
        return: a dictionary D
        """
        n = 10

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
            for p in range(len(self.paths[(s, t)])):
                U_new[(s, t)][p], U_old[(s, t)][p] = 0, 0
                D_new[(s, t)][p], D_old[(s, t)][p] = 0, 0

        for i in range(iterations):
            for e in self.dependencies:
                Q_new[e] = Q_old[e]
                temp = 0
                for (s, t) in self.dependencies[e]:
                    temp_st = 0
                    for p in self.dependencies[e][(s, t)]:
                        temp_st += pow(D_old[(s, t)][p], n)
                    temp += pow(temp_st, 1. / n)
                temp -= self.bandwidth[e]
                Q_new[e] += stepsize * self.plus(Q_old[e], temp)

            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in range(len(self.paths[(s, t)])):
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                R_new[(s, t)] += stepsize * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in range(len(self.paths[(s, t)])):
                    U_new[(s, t)][p] = U_old[(s, t)][p] + stepsize * self.plus(U_old[(s, t)][p],
                                                                               -D_old[(s, t)][p])

                    temp = 0
                    path = self.paths[(s, t)][p].path
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
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            '''Realize it using max function instead of approximation'''
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
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                obj += (D[(s, t)][p] - Y[(s, t)][p]) ** 2

        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve()
        logging.debug("status: " + problem.status)
        logging.debug("optimal values: " + str(problem.value))

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = D[(s, t)][p].value

        return D

    def alg(self, iterations, head, N1, N2, stepsize):
        Y = {}
        for (s, t) in self.paths:
            Y[(s, t)] = {}
            for p in range(len(self.paths[(s, t)])):
                Y[(s, t)][p] = 0

        feasibilities = []
        for t in range(iterations):
            Z = self.Estimate_Gradient(Y, head, N1, N2)
            self.adapt(Y, Z, 1. / (t + 1))
            if stepsize:
                Y = self.project_distributed(Y, 1000, stepsize)
                feasibility = self.feasibility(Y)
                feasibilities.append(feasibility)
            else:
                Y = self.project(Y)
            logging.debug(t)

        feasibility = self.feasibility(Y)
        feasibilities.append(feasibility)

        return Y, feasibilities


class MaxTP(Learning):
    def centralAlg(self):
        constr = []

        D = {}
        for (s, t) in self.paths:
            D[(s, t)] = {}
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            '''Realize it using max function instead of approximation'''
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
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                obj += D[(s, t)][p]

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        logging.debug("status: " + problem.status)
        logging.debug("optimal values: " + str(problem.value))

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = D[(s, t)][p].value

        feasibility = self.feasibility(D)
        return D, [feasibility]

    def distributedAlg(self, iterations, stepsize):
        n = 10

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
            for p in range(len(self.paths[(s, t)])):
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
                Q_new[e] += stepsize * self.plus(Q_old[e], temp)

            temp_R = {}
            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in range(len(self.paths[(s, t)])):
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                temp_R[(s, t)] = temp
                temp = np.expm1(temp)
                R_new[(s, t)] += stepsize * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in range(len(self.paths[(s, t)])):
                    U_new[(s, t)][p] = U_old[(s, t)][p] + stepsize * self.plus(U_old[(s, t)][p],
                                                                               np.expm1(-D_old[(s, t)][p]))

                    temp = 0
                    path = self.paths[(s, t)][p].path
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

                    D_new[(s, t)][p] = D_old[(s, t)][p] + stepsize * (
                            1 - temp - R_old[(s, t)] * np.exp(temp_R[(s, t)])
                            + U_old[(s, t)][p]) * np.exp(-D_old[(s, t)][p])

                U_old[(s, t)] = U_new[(s, t)].copy()
                D_old[(s, t)] = D_new[(s, t)].copy()
            Q_old, R_old = Q_new.copy(), R_new.copy()

        feasibility = self.feasibility(D_new)
        return D_new, [feasibility]

    def Lagrangian(self, D, Q, R, U, n):
        L = 0
        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L += D[(s, t)][p]

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = 0
                for p in self.dependencies[e][(s, t)]:
                    temp_st += pow(D[(s, t)][p], n)
                temp += pow(temp_st, 1. / n)
            temp -= self.bandwidth[e]
            temp = np.expm1(temp)
            L -= Q[e] * temp

        for (s, t) in self.paths:
            temp = 0
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            temp -= self.sourceRates[(s, t)]
            temp = np.expm1(temp)
            L -= R[(s, t)] * temp

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L -= U[(s, t)][p] * np.expm1(-D[(s, t)][p])
        return L


class MaxFair(Learning):
    def utility(self, x, alpha):
        if alpha == 1.0:
            return cp.log(x)
        else:
            return x ** (1 - alpha) / (1 - alpha)

    def centralAlg(self, alpha):
        constr = []

        D = {}
        for (s, t) in self.paths:
            D[(s, t)] = {}
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = cp.Variable()
                constr.append(D[(s, t)][p] >= 0)

        for e in self.dependencies:
            '''Realize it using max function instead of approximation'''
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
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            constr.append(temp <= self.sourceRates[(s, t)])

        obj = 0
        for l in self.learners:
            obj_l = 0
            for s in self.sources:
                (s, t, p) = self.slToStp[(s, l)]
                obj_l += D[(s, t)][p]
            obj += self.utility(obj_l, alpha)

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        logging.debug("status: " + problem.status)
        logging.debug("optimal values: " + str(problem.value))

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                D[(s, t)][p] = D[(s, t)][p].value

        feasibility = self.feasibility(D)
        return D, [feasibility]

    def distributedAlg(self, alpha, iterations, stepsize):
        n = 10

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
            for p in range(len(self.paths[(s, t)])):
                U_new[(s, t)][p], U_old[(s, t)][p] = 0, 0
                D_new[(s, t)][p], D_old[(s, t)][p] = 0, 0.1

        for i in range(iterations):
            for e in self.dependencies:
                Q_new[e] = Q_old[e]
                temp = 0
                for (s, t) in self.dependencies[e]:
                    temp_st = 0
                    for p in self.dependencies[e][(s, t)]:
                        temp_st += pow(D_old[(s, t)][p], n)
                    temp += pow(temp_st, 1. / n)
                temp -= self.bandwidth[e]
                Q_new[e] += stepsize * self.plus(Q_old[e], temp)

            for (s, t) in self.paths:
                R_new[(s, t)] = R_old[(s, t)]
                temp = 0
                for p in range(len(self.paths[(s, t)])):
                    temp += D_old[(s, t)][p]
                temp -= self.sourceRates[(s, t)]
                R_new[(s, t)] += stepsize * self.plus(R_old[(s, t)], temp)

            for (s, t) in self.paths:
                for p in range(len(self.paths[(s, t)])):
                    U_new[(s, t)][p] = U_old[(s, t)][p] + stepsize * self.plus(U_old[(s, t)][p],
                                                                               -D_old[(s, t)][p])

                    temp = 0
                    path = self.paths[(s, t)][p].path
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

                    temp_sum = 0
                    l = self.paths[(s, t)][p].learner
                    for source in self.sources:
                        temp_sum += self.getLearnerRate(D_old, source, l)

                    D_new[(s, t)][p] = D_old[(s, t)][p] + stepsize * (temp_sum ** (-alpha) - temp -
                                                                      R_old[(s, t)] + U_old[(s, t)][p])

                U_old[(s, t)] = U_new[(s, t)].copy()
                D_old[(s, t)] = D_new[(s, t)].copy()
            Q_old, R_old = Q_new.copy(), R_new.copy()

        feasibility = self.feasibility(D_new)
        return D_new, [feasibility]

    def Lagrangian(self, D, Q, R, U, n, alpha):
        L = 0
        for l in self.learners:
            obj_l = 0
            for s in self.sources:
                (s, t, p) = self.slToStp[(s, l)]
                obj_l += D[(s, t)][p]
            L += self.utility(obj_l, alpha)

        for e in self.dependencies:
            temp = 0
            for (s, t) in self.dependencies[e]:
                temp_st = 0
                for p in self.dependencies[e][(s, t)]:
                    temp_st += pow(D[(s, t)][p], n)
                temp += pow(temp_st, 1. / n)
            temp -= self.bandwidth[e]
            L -= Q[e] * temp

        for (s, t) in self.paths:
            temp = 0
            for p in range(len(self.paths[(s, t)])):
                temp += D[(s, t)][p]
            temp -= self.sourceRates[(s, t)]
            L -= R[(s, t)] * temp

        for (s, t) in self.paths:
            for p in range(len(self.paths[(s, t)])):
                L += U[(s, t)][p] * U[(s, t)][p]
        return L


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=5, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--stepsize', default=0.01, type=float, help="stepsize")
    parser.add_argument('--solver', type=str, help='solver type',
                        choices=['DFW', 'FW', 'DPGA', 'PGA', 'DMaxFair', 'MaxFair', 'DMaxTP', 'MaxTP'])

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='DEBUG', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    np.random.seed(args.random_seed + 2023)

    fname = 'Problem_10/Problem_{}_{}learners_{}sources_{}types'.format(
        args.graph_type, args.learners, args.sources, args.types)
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)

    if args.solver == 'DFW':
        alg = FrankWolf(P)
        t1 = time.time()
        Y, feasibilities = alg.alg(iterations=50, head=15, N1=50, N2=50, stepsize=args.stepsize)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'FW':
        alg = FrankWolf(P)
        t1 = time.time()
        Y, feasibilities = alg.alg(iterations=50, head=15, N1=50, N2=50, stepsize=0)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'DPGA':
        alg = ProjectAscent(P)
        t1 = time.time()
        Y, feasibilities = alg.alg(iterations=50, head=15, N1=50, N2=50, stepsize=args.stepsize)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'PGA':
        alg = ProjectAscent(P)
        t1 = time.time()
        Y, feasibilities = alg.alg(iterations=50, head=15, N1=50, N2=50, stepsize=0)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'DMaxTP':
        alg = MaxTP(P)
        t1 = time.time()
        Y, feasibilities = alg.distributedAlg(iterations=1000, stepsize=args.stepsize)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'MaxTP':
        alg = MaxTP(P)
        t1 = time.time()
        Y, feasibilities = alg.centralAlg()
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'DMaxFair':
        alg = MaxFair(P)
        t1 = time.time()
        Y, feasibilities = alg.distributedAlg(alpha=2, iterations=1000, stepsize=args.stepsize)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    if args.solver == 'MaxFair':
        alg = MaxFair(P)
        t1 = time.time()
        Y, feasibilities = alg.centralAlg(alpha=2)
        t2 = time.time()
        obj = alg.objU(Y=Y, N1=100, N2=100)
        period = t2 - t1
        print(t2 - t1, Y, obj, feasibilities[-1])

    fname = 'Result_15_{}/Result_{}_{}learners_{}sources_{}types_{}stepsize'.format(
        args.solver, args.graph_type, args.learners, args.sources, args.types, args.stepsize)
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump((period, Y, obj, feasibilities), f)
