import numpy as np


def tuplelist(ps):
    return [tuple(e) for e in ps]


def keep_unique(ps):
    return sorted(set([tuple(e) for e in ps]))


def build_nanoparticle(n=5):
    '''
    Builds a pointed octahedral nanoparticle with 2*n-1 layers.
    All coordinates are integral.
    '''

    qs = np.array([(1, 0, 1), (-1, 0, 1), (0, -1, 1), (0, 1, 1)])

    layer = np.array([(0, 0, -n)])
    ps = tuplelist(layer)
    for i in range(n):
        next = keep_unique([p + q for q in qs for p in layer])
        ps += next
        layer = next

    ps = np.array(ps)
    ps = keep_unique(np.concatenate((ps, -ps)))
    ps = np.array(ps)
    norms = np.linalg.norm(ps, axis=1)
    indices = np.argsort(norms)[:-6]
    return ps[indices]


def get_neighbours(ps):

    threshold = (np.sqrt(2) + 2) / 2

    bonds = []
    coord = []
    for i, p in enumerate(ps):
        d = np.linalg.norm(ps - p, axis=1)
        indices = np.where(d < threshold)[0]
        num_nbrs = len(set(indices)) - 1
        coord += [num_nbrs]
        bonds += [(i, j) for j in indices if j > i]

    return coord, bonds


def get_layers(ps):

    zs = ps[:, 2]
    return [np.where(zs == z)[0] for z in np.unique(zs)]


def get_partitions(ps):

    tolerance = 1E-4
    px0 = ps[:, 0] < +tolerance
    px1 = ps[:, 0] > -tolerance

    py0 = ps[:, 1] < +tolerance
    py1 = ps[:, 1] > -tolerance

    pz0 = ps[:, 2] < +tolerance
    pz1 = ps[:, 2] > -tolerance

    return [(px0, px1), (py0, py1), (pz0, pz1)]


def get_attributes(ps):

    coordination, bonds = get_neighbours(ps)

    labels = {  6: 'corner',
                7: 'edge',
                9: 'terrace',
               12: 'interior'}

    site_types = [labels[c] for c in coordination]
    return site_types, bonds, get_layers(ps), get_partitions(ps)
