from networkx import DiGraph, shortest_path
import networkx
import pickle
import topologies
import numpy as np
import logging, argparse
import random


class Problem:
    def __init__(self, sourceRates, sources, learners, bandwidth, G, paths, prior, T, slToStp, sourceParameters):
        self.sourceRates = sourceRates
        self.sources = sources
        self.learners = learners
        self.bandwidth = bandwidth
        self.G = G
        self.paths = paths
        self.prior = prior
        self.T = T
        self.slToStp = slToStp
        self.sourceParameters = sourceParameters


class Path:
    def __init__(self, G, source, learner):
        self.learner = learner
        self.path = shortest_path(G, source, learner, 'weight')

    # def edgeIsInPath(self, first_node, second_node):
    #     if first_node not in self.path:
    #         return False
    #     i = self.path.index(first_node)
    #     if i + 1 == len(self.path):
    #         return False
    #     else:
    #         return second_node == self.path[i + 1]


def main():
    parser = argparse.ArgumentParser(description='Simulate a Network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--min_bandwidth', default=10, type=float, help='Minimum bandwidth of each edge')
    parser.add_argument('--max_bandwidth', default=20, type=float, help="Maximum bandwidth of each edge")

    parser.add_argument('--min_datarate', default=2, type=float, help='Minimum data rate of each item at each sources')
    parser.add_argument('--max_datarate', default=5, type=float, help="Maximum bandwidth of each edge")

    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=5, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--dimension', default=100, type=int, help='Feature dimension')

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--graph_degree', default=4, type=int,
                        help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p', default=0.10, type=int, help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--T', default=1, type=float, help="Duration of experiment")

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)

    def graphGenerator():
        if args.graph_type == "erdos_renyi":
            return networkx.erdos_renyi_graph(args.graph_size, args.graph_p)
        if args.graph_type == "balanced_tree":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(args.graph_degree)))
            return networkx.balanced_tree(args.graph_degree, ndim)
        if args.graph_type == "cicular_ladder":
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.circular_ladder_graph(ndim)
        if args.graph_type == "cycle":
            return networkx.cycle_graph(args.graph_size)
        if args.graph_type == 'grid_2d':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim, ndim)
        if args.graph_type == 'lollipop':
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.lollipop_graph(ndim, ndim)
        if args.graph_type == 'expander':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.margulis_gabber_galil_graph(ndim)
        if args.graph_type == "hypercube":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(2.0)))
            return networkx.hypercube_graph(ndim)
        if args.graph_type == "star":
            ndim = args.graph_size - 1
            return networkx.star_graph(ndim)
        if args.graph_type == 'barabasi_albert':
            return networkx.barabasi_albert_graph(args.graph_size, args.graph_degree)
        if args.graph_type == 'watts_strogatz':
            return networkx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
        if args.graph_type == 'regular':
            return networkx.random_regular_graph(args.graph_degree, args.graph_size)
        if args.graph_type == 'powerlaw_tree':
            return networkx.random_powerlaw_tree(args.graph_size)
        if args.graph_type == 'small_world':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.navigable_small_world_graph(ndim)
        if args.graph_type == 'geant':
            return topologies.GEANT()
        if args.graph_type == 'dtelekom':
            return topologies.Dtelekom()
        if args.graph_type == 'abilene':
            return topologies.Abilene()
        if args.graph_type == 'servicenetwork':
            return topologies.ServiceNetwork()

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2023)

    logging.info('Generating ' + args.graph_type + ' graph')
    temp_graph = graphGenerator()  # use networkx to generate a graph
    # networkx.draw(temp_graph)
    # plt.draw()
    V = len(temp_graph.nodes())
    E = len(temp_graph.edges())
    logging.debug('nodes: ' + str(temp_graph.nodes()))  # list
    logging.debug('edges: ' + str(temp_graph.edges()))  # list of node pair
    G = DiGraph()  # generate a DiGraph

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())  # add node from temp_graph to G
    for (x, y) in temp_graph.edges():  # add edge from temp_graph to G
        xx = number_map[x]
        yy = number_map[y]
        G.add_edges_from(((xx, yy), (yy, xx)))
    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size, edge_size))
    logging.debug('G is:' + str(G.nodes()) + str(G.edges()))

    logging.info('Generating sources')
    sources = random.sample(range(graph_size), args.sources)

    logging.info('Generating learners')
    learners = random.sample(range(graph_size), args.learners)

    logging.info('Generating bandwidth')
    bandwidth = {}
    for e in G.edges():
        bandwidth[e] = random.uniform(args.min_bandwidth, args.max_bandwidth)

    logging.info('Generating types')
    if args.learners <= args.types:
        types = list(range(args.learners))
    if args.learners > args.types:
        types = list(range(args.types))
        types += random.choices(range(args.types), k=args.learners - args.types)
    # types = random.choices(range(args.types), k=args.learners)
    l_t_tuple = list(zip(learners, types))
    l_t_dict = dict(l_t_tuple)

    # a dictionary for types: key is the type, value is a list of learners with this type
    types = {}
    for (l, t) in l_t_tuple:
        if t in types:
            types[t].append(l)
        else:
            types[t] = [l]

    logging.info('Generating prior')
    prior = {}
    prior['noice'] = {}
    prior['cov'] = {}
    prior['beta'] = {}
    i = 0
    valid_dimension = int(np.floor(args.dimension / args.learners))
    noices = {}
    for t in types:
        for s in sources:
            noices[(t, s)] = np.random.uniform(0.5, 1)
    for l in learners:
        diag = np.random.uniform(0, 0.01, args.dimension)
        diag[i * valid_dimension:(i + 1) * valid_dimension] = np.random.uniform(100, 200, valid_dimension)
        prior['cov'][l] = np.diag(diag)
        prior['noice'][l] = {}
        for s in sources:
            prior['noice'][l][s] = noices[(l_t_dict[l], s)]
        prior['beta'][l] = np.zeros((args.dimension, 1))
        prior['beta'][l][i * valid_dimension:(i + 1) * valid_dimension] = np.ones((valid_dimension, 1))
        i += 1

    logging.info('Generating source rates, paths and parameters of data')
    sourceRates = {}
    sourceParameters = {}
    sourceParameters['mean'] = {}
    sourceParameters['cov'] = {}
    paths = {}
    slToStp = {}
    valid_dimension = int(np.floor(args.dimension / args.sources))
    i = 0
    for s in sources:
        diag = np.zeros(args.dimension)
        diag[i * valid_dimension:(i + 1) * valid_dimension] = np.random.uniform(1, 2, valid_dimension)
        sourceParameters['cov'][s] = np.diag(diag)
        sourceParameters['mean'][s] = np.zeros(args.dimension)
        for t in types:
            sourceRates[(s, t)] = np.random.uniform(args.min_datarate, args.max_datarate)
            for l in types[t]:
                if (s, t) in paths:
                    paths[(s, t)].append(Path(G, s, l))
                else:
                    paths[(s, t)] = [Path(G, s, l)]
                slToStp[(s, l)] = (s, t, len(paths[(s, t)]) - 1)
        i += 1

    P = Problem(sourceRates, sources, learners, bandwidth, G, paths, prior, args.T, slToStp, sourceParameters)
    fname = 'Problem/Problem_' + args.graph_type
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(P, f)


if __name__ == '__main__':
    main()
