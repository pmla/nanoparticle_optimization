import numpy as np


def to_rotation_matrix(q):

    a, b, c, d = q

    u = np.zeros((3, 3)).astype(np.double)

    u[0, 0] = a * a + b * b - c * c - d * d
    u[0, 1] = 2 * b * c - 2 * a * d
    u[0, 2] = 2 * b * d + 2 * a * c

    u[1, 0] = 2 * b * c + 2 * a * d
    u[1, 1] = a * a - b * b + c * c - d * d
    u[1, 2] = 2 * c * d - 2 * a * b

    u[2, 0] = 2 * b * d - 2 * a * c
    u[2, 1] = 2 * c * d + 2 * a * b
    u[2, 2] = a * a - b * b - c * c + d * d

    return u


qgenerator_laue_O = [[1.0, 0, 0, 0],
                     [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                     [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0],
                     [0, 0, 0, 1.0],
                     [0.5, 0.5, -0.5, 0.5],
                     [np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2],
                     [0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2],
                     [0.5, 0.5, 0.5, -0.5],
                     [0, 1.0, 0, 0],
                     [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2],
                     [0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                     [0.5, 0.5, -0.5, -0.5],
                     [0, -np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                     [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0, 0],
                     [0, 0, 1.0, 0],
                     [0.5, -0.5, -0.5, -0.5],
                     [0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                     [0.5, -0.5, 0.5, 0.5],
                     [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0],
                     [0.5, -0.5, 0.5, -0.5],
                     [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
                     [0.5, -0.5, -0.5, 0.5],
                     [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                     [0.5, 0.5, 0.5, 0.5]]

norms = np.linalg.norm(qgenerator_laue_O, axis=1)
assert(np.linalg.norm(norms - 1) < 1E-9)

generator_laue_O = np.array([to_rotation_matrix(q) for q in qgenerator_laue_O])
