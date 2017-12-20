from __future__ import print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
from gurobipy import *
import initialize_nanoparticle
import annealed
import automorphisms


def build_mipmodel(intercept, clusters, numb):

    xclusters = [(sites[0], param) for (sites, param) in clusters
                 if len(sites) == 1]
    yclusters = [(sites, param) for (sites, param) in clusters
                 if len(sites) >= 2]
    xindices = sorted([i for i, param in xclusters])

    model = Model("mip1")
    x = {site: model.addVar(vtype=GRB.BINARY) for site, param in xclusters}
    y = {sites: model.addVar(vtype=GRB.BINARY) for sites, param in yclusters}
    model.update()

    for (sites, param) in yclusters:

        if param >= 0:
            model.addConstr( y[sites] >= sum([x[e] for e in sites])
                                         - len(sites) + 1 )
        else:
            for e in sites:
                model.addConstr( y[sites] <= x[e] )

    model.addConstr( sum(x.values()) == numb )

    model.setObjective( intercept
                + sum([param * x[site] for (site, param) in xclusters])
                + sum([param * y[sites] for (sites, param) in yclusters]),
                GRB.MINIMIZE)

    model.params.MIPGap = 0
    model.params.Presolve = 2
    model.params.MIPFocus = 2
    return model, x, xindices

def solve_mipmodel(numb, intercept, clusters, sym_mappings, num_solutions):

    model, x, xindices = build_mipmodel(intercept, clusters, numb)

    energies = []
    for it in range(num_solutions):

        model.optimize()
        status = model.getAttr("Status")
        if status not in [GRB.OPTIMAL, GRB.USER_OBJ_LIMIT]:
            break

        objective = model.getAttr("ObjVal")
        xs = np.array([int(round(x[e].x)) for e in xindices]).astype(np.int8)
        xprev = xs

        energies += [objective]

        #sum {xj : x'_j = 0} + sum {1-xj : x'_j = 1} >= 1
        for mapping in sym_mappings:
            inactive = [x[mapping[i]] for i in xindices if xprev[i] == 0]
            active = [x[mapping[i]] for i in xindices if xprev[i] == 1]
            model.addConstr(sum(inactive + [1 - e for e in active]) >= 1)
        model.update()
        model.params.BestObjStop = objective + 1E-9
        model.params.MIPFocus=0

    return energies

def calc_formation_energy(num_atoms, y0, y1, xs, ys):

    fracs = np.array(xs).astype(np.double) / num_atoms
    ys = np.array(ys).astype(np.double)
    ys = ys - (y0 + (y1 - y0) * fracs)
    return ys / num_atoms

def main(num_solutions):

    coords, intercept, clusters = initialize_nanoparticle.get_params()
    _, sym_mappings = automorphisms.gen_automorphisms(coords)
    num_atoms = len(coords)

    mip_energies = []
    ns = np.arange(0, num_atoms + 1)
    for numb in ns:
        energies = solve_mipmodel(numb, intercept, clusters,
                                   sym_mappings, num_solutions)
        mip_energies += [energies]

    E0 = mip_energies[0][0]
    E1 = mip_energies[-1][0]

    ans, aenergies = annealed.get_annealed_configurations(intercept, clusters)
    afs = calc_formation_energy(num_atoms, E0, E1, ans, aenergies)
    plt.scatter(ans, afs, marker='x', c='C1', lw=2,
                label='simulated annealing')

    ground_data = []
    nth_data = []
    for i in range(num_solutions):

        _ns = []
        energies = []
        for n, ys in zip(ns, mip_energies):
            if i < len(ys):
                _ns += [n]
                energies += [ys[i]]

        fs = calc_formation_energy(num_atoms, E0, E1, _ns, energies)
        if i == 0:
            ground_data += zip(_ns, list(fs))
        else:
            nth_data += zip(_ns, list(fs))

    ns, fs = zip(*ground_data)
    plt.scatter(ns, fs, marker='o', edgecolor='C0',
                facecolor='none', lw=2, label='MIP ground state', zorder=1)

    if len(nth_data):
        ns, fs = zip(*nth_data)
        plt.scatter(ns, fs, marker='o', edgecolor='C2', facecolor='none',
                    lw=2, label='MIP nth energy level state', zorder=0)

    plt.xlabel('Number of Pt atoms')
    plt.ylabel('$E_f$ (meV / atom)')
    plt.legend(loc=9)
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise Exception("usage: optimize.py [num_solutions]")

    num_solutions = int(sys.argv[1])
    main(num_solutions)

