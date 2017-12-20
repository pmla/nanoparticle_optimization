import numpy as np


def evaluate(xs, intercept, clusters):

    E = intercept

    for sites, param in clusters:
        if all([xs[i] for i in sites]):
            E += param
    return E


def read_annealing_configs(path):

    lines = open(path).read().split('\n')
    lines = [e for e in lines if len(e) > 0]

    sizes = [int(e[2:]) for i, e in enumerate(lines) if i % 2 == 0]
    configs = [[int(e) for e in line.split()]
               for i, line in enumerate(lines) if i % 2 == 1]
    configs = [e for e in configs]
    return dict([(s, c) for s, c in zip(sizes, configs)])


def get_annealed_configurations(intercept, clusters):
    configs = read_annealing_configs('annealing_configurations.txt')

    data = [(n, evaluate(config, intercept, clusters))
            for n, config in configs.items()]
    ns, energies = zip(*sorted(data))
    return np.array(ns), np.array(energies)
