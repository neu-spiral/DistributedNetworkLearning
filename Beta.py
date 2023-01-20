import logging, argparse
import pickle
import numpy as np
from ProbGenerate import Problem, Path
from distributedSolver import Learning

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate model through MAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="geant", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample2'])
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--solver', type=str, help='solver type',
                        choices=['DFW', 'FW', 'DPGA', 'PGA', 'DMaxFair', 'MaxFair', 'DMaxTP', 'MaxTP'])
    parser.add_argument('--stepsize', default=0.01, type=float, help="stepsize for FW")

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2023)
    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    fname = 'Result_15_{}/Result_{}_{}stepsize'.format(args.solver, args.graph_type, args.stepsize)

    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    fname = "Problem/Problem_{}".format(args.graph_type)
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)

    learning = Learning(P)
    beta = P.prior['beta']
    covariance = P.prior['cov']
    learners = P.learners
    T = P.T

    N1 = 100
    N2 = 100
    result = results[1]
    dist = 0
    if result == 0:
        distance = 0
    for l in learners:
        noice = P.prior['noice'][l]
        norm = 0
        for j in range(N1):
            n = learning.generate_sample1(result, l)
            for i in range(N2):
                features = learning.generate_sample2(n)
                a = 0
                b = 0
                for s in features:
                    for feature in features[s]:
                        a += np.dot(feature, feature.transpose()) / noice[s]
                        y = np.dot(feature.transpose(), beta[l]) + np.random.normal(0, noice[s])
                        b += feature * y / noice[s]
                temp1 = a + np.linalg.inv(covariance[l])
                temp1 = np.linalg.inv(temp1)
                temp2 = np.dot(np.linalg.inv(covariance[l]), beta[l]) + b
                map_l = np.dot(temp1, temp2)

                # norm_temp = np.linalg.norm(map_l - beta[l])
                # if norm_temp > 1:
                #     print(norm_temp)
                norm += np.linalg.norm(map_l - beta[l])
        norm = norm / N1 / N2
        dist += norm
    distance = dist / len(learners)
    print(distance)
    fname = 'Result_15_{}/beta_{}_{}stepsize'.format(args.solver, args.graph_type, args.stepsize)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(distance, f)
