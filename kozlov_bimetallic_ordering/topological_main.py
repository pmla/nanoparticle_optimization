from __future__ import print_function
import numpy as np
from gurobipy import *
import octahedron
import energetic_data
import draw_nanoparticle


def mip_optimize(params, site_types, bonds, layers, partitions, numb):

    xindices = list(range(len(site_types)))

    model = Model("nanoparticle")
    x = {i: model.addVar(vtype=GRB.BINARY) for i in xindices}
    y = {b: model.addVar(vtype=GRB.BINARY) for b in bonds}

    bond_energy_AB = params['bond']
    w = np.zeros(len(xindices)).astype(np.double)
    for b in bonds:

        i, j = b
        if bond_energy_AB <= 0:
            model.addConstr(y[b] >= x[i] + x[j] - 1)
        else:
            model.addConstr(y[b] <= x[i])
            model.addConstr(y[b] <= x[j])

        w[i] += bond_energy_AB
        w[j] += bond_energy_AB

    model.addConstr(sum(x.values()) == numb)

    site_energies = [params[site_types[i]] for i in xindices]
    layer_energy = params['layer']
    layer_contribution = model.addVar(vtype=GRB.CONTINUOUS,
                                      lb=-float("inf"), ub=float("inf"))

    if layer_energy == 0:
        layer_contribution.lb = 0
        layer_contribution.ub = 0
    else:
        layer_indices = range(len(layers))

        decision = {i: model.addVar(vtype=GRB.BINARY) for i in layer_indices}
        deltap = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                  for i in layer_indices}
        deltam = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                  for i in layer_indices}

        for i, layer in enumerate(layers):
            model.addConstr(deltap[i] - deltam[i] ==
                            2 * sum([x[j] for j in layer]) - len(layer))
            model.addConstr(deltap[i] <= decision[i] * 2 * len(layer))
            model.addConstr(deltam[i] <= (1 - decision[i]) * 2 * len(layer))

        model.addConstr(layer_contribution == layer_energy * sum([deltap[i] + deltam[i] for i in layer_indices]))

    # symmetry-breaking constraints
    for pa, pb in partitions:
        model.addConstr(sum([x[e] for e in pa]) >= sum([x[e] for e in pb]))

    model.setObjective(sum([(site_energies[i] + w[i]) * x[i] for i in xindices])
                       - 2 * bond_energy_AB * sum(y.values())
                       + layer_contribution, GRB.MINIMIZE)

    model.params.MIPGap = 0
    model.params.Presolve = 2
    model.params.MIPFocus = 2
    model.optimize()
    status = model.getAttr("Status")
    print("status:", status)
    assert(status == GRB.OPTIMAL)

    objective = model.getAttr("ObjVal")
    print("objective:", objective)

    xs = [int(round(x[e].x)) for e in xindices]
    return objective, np.array(xs)


def run(size, element):

    coords = octahedron.build_nanoparticle(size)
    site_types, bonds, layers, partitions = octahedron.get_attributes(coords)

    num_sites = len(coords)
    numb = num_sites // 2
    numa = num_sites - numb
    print("num. sites:", num_sites)
    print("Pd%d%s%d" % (numa, element, numb))

    params = energetic_data.parameters[element]
    objective, xs = mip_optimize(params, site_types, bonds,
                                 layers, partitions, numb)

    print("energy:", objective)
    print("configuration:", xs)
    draw_nanoparticle.draw(coords, xs)


def main():

    if len(sys.argv) != 3:
        raise Exception("arguments: [element] [size]")

    element = sys.argv[1]
    size = int(sys.argv[2])

    if size <= 0:
        raise Exception("size must be a positive integer")

    if element not in ['Ag', 'Au', 'Cu', 'Zn']:
        raise Exception("element must be one of {Ag, Au, Cu, Zn}")

    run(size, element)


if __name__ == "__main__":
    main()
