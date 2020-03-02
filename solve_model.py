import os
import pickle
import argparse
import numpy as np
from ase.db import connect
from ase.visualize import view
from gurobipy import Model, GRB
from build_model import build_cluster_space, count_clusters, fit_ecis


def solve_mipmodel(intercept, clusters, num=-1):
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
            model.addConstr(y[sites] >= sum([x[e] for e in sites]) - len(sites) + 1)
        else:
            for e in sites:
                model.addConstr(y[sites] <= x[e])

    if num != -1:
        model.addConstr(sum(x.values()) == num)

    model.setObjective(intercept
                       + sum([param * x[site] for (site, param) in xclusters])
                       + sum([param * y[sites] for (sites, param) in yclusters]),
                       GRB.MINIMIZE)

    model.params.MIPGap = 0
    model.params.Presolve = 2
    #model.params.MIPFocus = 2
    model.params.Threads = 2

    model.optimize()
    #status = model.getAttr("Status")
    #if status not in [GRB.OPTIMAL, GRB.USER_OBJ_LIMIT]:
    #    pass

    objective = model.getAttr("ObjVal")
    xs = np.array([int(round(x[e].x)) for e in xindices])

    print(objective, np.sum(xs))
    return objective, xs


def run(path):
    path = path[0]
    if not os.path.exists(path):
        raise ValueError("Path does not exist")

    rows = list(connect(path).select())
    atoms = rows[0].toatoms()
    unique_numbers = np.unique([row.numbers for row in rows])
    if len(unique_numbers) != 2:
        raise ValueError("Database does not contain binary structures.")
    zA, zB = unique_numbers

    build_cluster_space(atoms)
    count_clusters(path, zA, zB)
    intercept, ecis = fit_ecis(len(atoms), num_features=70)

    cluster_space = pickle.load(open('cluster_space.pickle', 'rb'))
    clusters = []
    for param, (k, cluster) in zip(ecis, sorted(cluster_space.items())):
        for sites in cluster:
            clusters.append((sites, param))
    objective, xs = solve_mipmodel(intercept, clusters, num=144)

    atoms.numbers = np.array([zA, zB])[xs]
    view(atoms)


parser = argparse.ArgumentParser(description='Build a database of symmetric '
                                             'nanoparticle configurations')
parser.add_argument('path', type=str, nargs='+', help='Path to database')
args = parser.parse_args()
run(args.path)
