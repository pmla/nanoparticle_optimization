import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed
from ase.db import connect
from ase.data import chemical_symbols, atomic_numbers
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.cluster.icosahedron import Icosahedron
from structure_analyzer import build_neighbor_graph, get_site_types


def relax_structure(atoms, fmax=0.01):
    atoms.set_calculator(EMT())
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    return atoms, atoms.get_potential_energy()


def worker(i, db, site_types, atoms, c):
    c = np.array(c)
    atoms = atoms.copy()
    atoms.numbers = c[site_types]

    atoms, E = relax_structure(atoms)
    db.write(atoms, EMT_energy=E)
    print(i, E)


def run(A, B, num_jobs):
    A = A[0]
    B = B[0]
    num_jobs = num_jobs[0]
    for s in [A, B]:
        if s not in chemical_symbols:
            raise ValueError("{0} is not a valid chemical species.".format(s))

    zA = atomic_numbers[A]
    zB = atomic_numbers[B]
    (zA, A), (zB, B) = sorted([(zA, A), (zB, B)])

    atoms = Icosahedron(A, 5)
    graph = build_neighbor_graph(atoms, cutoff=3.7)
    order, site_types = get_site_types(graph)
    atoms, _ = relax_structure(atoms)

    m = len(np.unique(site_types))
    cs = list(itertools.product([zA, zB], repeat=m))

    db = connect('symmetric_{0}{1}.db'.format(A, B), append=False)
    Parallel(n_jobs=num_jobs)(delayed(worker)(i, db, site_types, atoms, c)
                              for i, c in enumerate(cs))


parser = argparse.ArgumentParser(description='Build a database of symmetric '
                                             'nanoparticle configurations')
parser.add_argument('A', type=str, nargs='+', help='First chemical species')
parser.add_argument('B', type=str, nargs='+', help='Second chemical species')
parser.add_argument('ncpus', type=int, nargs='+', help='Number of cpus to use')
args = parser.parse_args()
run(args.A, args.B, args.ncpus)
