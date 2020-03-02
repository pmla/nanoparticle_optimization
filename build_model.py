import pickle
import numpy as np
from ase.db import connect
from structure_analyzer import (build_neighbor_graph,
                                get_site_types,
                                build_clusterspace)
import bestsubset


def build_cluster_space(atoms):
    graph = build_neighbor_graph(atoms, cutoff=3)
    order, site_types = get_site_types(graph)
    cluster_space = build_clusterspace(graph, site_types, [0, 1, 4, 2])

    print("order:", order)
    print("site types:", np.bincount(site_types))
    print("cluster space:")
    for (order, key), v in sorted(cluster_space.items()):
        print('\t', order, len(v))

    pickle.dump(cluster_space, open('cluster_space.pickle', 'wb'))


def count_clusters(path, zA, zB):
    cluster_space = pickle.load(open('cluster_space.pickle', 'rb'))
    rows = list(connect(path).select())
    species_key = {zA: 0, zB: 1}

    keys = sorted(cluster_space)
    orbits = [np.array(cluster_space[key]) for key in keys]
    A = []
    counts = []
    energies = []
    for row in rows:
        numbers = np.array([species_key[z] for z in row.numbers])
        count = list(numbers).count(1)
        energy = row['EMT_energy']
        counts.append(count)
        energies.append(energy)
        print(count, energy)

        arow = []
        for orbit in orbits:
            sums = np.sum(numbers[orbit], axis=1)
            cluster_size = len(orbit[0])
            num_active = np.sum(sums == cluster_size)
            arow.append(num_active)
        A.append(arow)
    A = np.array(A)
    cs = np.array(counts)
    Es = np.array(energies)
    np.save('A.npy', A)

    i0 = np.where(cs == 0)[0][0]
    i1 = np.where(cs == row.natoms)[0][0]
    E0 = Es[i0]
    E1 = Es[i1]
    fs = cs / cs[i1]
    ys = (Es - E0) - (E1 - E0) * fs
    np.save('b.npy', ys)


def fit_ecis(natoms, num_features):
    A = np.load('A.npy')
    b = np.load('b.npy')
    indices0 = np.where(b <= 0)[0]
    indices1 = np.where(b > 0)[0]
    np.random.shuffle(indices1)
    indices = np.concatenate((indices0, indices1[:len(indices0) // 4]))
    A = A[indices].astype(np.float)
    b = b[indices]

    # Add intercept 'feature'
    A = np.array([[1] + list(row) for row in A])

    result = bestsubset.solve_greedy(A, b, max_features=num_features)
    indices, weights = result
    x = weights[:, num_features - 1]

    delta = A @ x - b
    rmse = np.sqrt(np.mean(delta**2)) * 1000 / natoms
    print("RMSE: %.3f meV / atom" % rmse)
    print(A.shape)
    return x[0], x[1:]
