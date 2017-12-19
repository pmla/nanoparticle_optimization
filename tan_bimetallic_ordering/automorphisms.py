import numpy as np
import scipy.spatial
import scipy.optimize
import symmetries


def gen_automorphisms(pos):

    sym = [np.dot(pos, g) for g in symmetries.generator_laue_O]
    sym += [np.dot(pos, -g) for g in symmetries.generator_laue_O]
    sym = np.array(sym)

    full_auto = []
    for s in sym:
        dist = scipy.spatial.distance.cdist(pos, s)
        res = scipy.optimize.linear_sum_assignment(dist)[1]
        full_auto += [res]
    full_auto = np.array(full_auto).T

    auto = [np.unique(e) for e in full_auto]
    return auto, np.array(full_auto).T

