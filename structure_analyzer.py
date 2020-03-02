import networkx
import itertools
import collections
import numpy as np
from scipy.spatial.distance import cdist
from ase.geometry.dimensionality.disjoint_set import DisjointSet


def build_neighbor_graph(atoms, cutoff=3.7):
    positions = atoms.get_positions()

    # Create edges as pairs of atoms whose distance is smaller than cutoff.
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, float("inf"))
    edges = np.sort(list(zip(*np.where(distances < cutoff))))
    edges = np.unique(edges, axis=0)

    # Create a graph using the edges
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph


def get_site_types(graph):
    # Find graph automorphisms
    matcher = networkx.algorithms.isomorphism.GraphMatcher(graph, graph)
    automorphisms = list(matcher.isomorphisms_iter())

    # Use a disjoint set (union-find) to merge equivalent sites
    uf = DisjointSet(len(graph))
    for automorphism in automorphisms:
        for i, j in sorted(automorphism.items()):
            uf.merge(i, j)

    order = len(automorphisms)
    return order, uf.get_components(relabel=True)


def canonical_form(site_types, distances, nodes):
    best = None
    for perm in itertools.permutations(nodes):
        m = [[(site_types[i], distances[i][j]) for j in perm] for i in perm]
        m = tuple([tuple(e) for e in m])
        if best is None:
            best = m
        best = max(best, m)
    return (len(best), best)


def build_clusterspace(graph, site_types, cutoffs):
    distances = dict(networkx.all_pairs_shortest_path_length(graph))

    num_atoms = len(graph)
    cs = collections.defaultdict(list)
    for n in range(1, len(cutoffs)):
        for nodes in itertools.combinations(range(num_atoms), n):
            if all([distances[i][j] <= cutoffs[n]
                    for i, j in itertools.combinations(nodes, 2)]):
                key = canonical_form(site_types, distances, nodes)
                cs[key].append(nodes)
    return cs
