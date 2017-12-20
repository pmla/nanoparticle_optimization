import numpy as np
import scipy.spatial
import itertools


def read_coefficients(path):

    lines = open(path).read().split('\n')
    lines = [e for e in lines if len(e) > 0]
    sizes = [int(e) for i, e in enumerate(lines) if i % 4 == 0]
    weights = [float(e) for i, e in enumerate(lines) if i % 4 == 2]

    weight0 = [w for s, w in zip(sizes, weights) if s == 0][0]
    weights1 = [w for s, w in zip(sizes, weights) if s == 1]
    weights2 = [w for s, w in zip(sizes, weights) if s == 2]
    weights3 = [w for s, w in zip(sizes, weights) if s == 3]
    return weight0, weights1 + weights2 + weights3


def read_integral_coordinates(path):

    lines = open(path).read().split('\n')
    coords = [[int(e) for e in line.split()] for line in lines]
    return np.array(coords)


def get_unique_clusters():

    n1a = [(0,), (7,), (17,), (30,), (39,)]
    n2b = [(7, 5), (39, 21), (42, 37), (29, 11),
           (15, 33), (50, 8), (17, 10), (0, 12)]
    n2c = [(47, 53), (17, 30), (37, 5), (0, 15), (12, 8)]
    n3d = [(1, 3, 14), (26, 45, 38), (29, 11, 50), (52, 53, 54),
           (7, 42, 37), (47, 17, 10), (51, 12, 9)]
    n3e = [(7, 5, 2), (9, 0, 3)]
    n3f = [(30, 53, 49), (47, 23, 17), (8, 45, 50),
           (33, 39, 15), (11, 18, 29), (7, 44, 51)]

    n1 = n1a
    n2 = n2b + n2c
    n3 = n3d + n3e + n3f
    return n1, n2, n3


def gen_fingerprints(points):

    distances = scipy.spatial.distance.cdist(points, points)
    sig = np.sort(distances)
    return [tuple(e) for e in sig], distances


def gen_sig(distances, site_types, sites):

    sig = []
    for i in sites:
        ds = sorted([distances[i, j] for j in sites if j != i])
        line = tuple([site_types[i]] + ds)
        sig += [line]
    return tuple(sorted(sig))


def gen_site_types(coords, n1):

    num_atoms = len(coords)
    fp, distances = gen_fingerprints(coords)

    site_types = []
    for i in range(num_atoms):
        for index, c in enumerate(n1):
            if fp[i] == fp[c[0]]:
                site_types += [index]
    assert(len(site_types) == num_atoms)
    return site_types, distances


def get_cluster_instances(coords, n1, n2, n3, weights):

    num_atoms = len(coords)
    site_types, distances = gen_site_types(coords, n1)

    target_sigs = [gen_sig(distances, site_types, sites)
                   for sites in n1 + n2 + n3]
    assert len(target_sigs) == len(set(target_sigs))

    singlets = itertools.combinations(range(num_atoms), 1)
    doublets = itertools.combinations(range(num_atoms), 2)
    triplets = itertools.combinations(range(num_atoms), 3)
    all_sites = itertools.chain(singlets, doublets, triplets)

    clusters = []
    for sites in all_sites:
        sig0 = gen_sig(distances, site_types, sites)
        try:
            index = target_sigs.index(sig0)
            clusters += [(sites, weights[index])]
        except ValueError:
            pass

    return clusters


def get_params():

    intercept, weights = read_coefficients('cluster_coefficients.txt')
    coords = read_integral_coordinates('rounded_coordinates.txt')

    n1, n2, n3 = get_unique_clusters()
    clusters = get_cluster_instances(coords, n1, n2, n3, weights)

    return coords, intercept, clusters
