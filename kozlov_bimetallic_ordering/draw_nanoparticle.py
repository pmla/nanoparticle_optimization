import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw(points, config):

    fig = plt.figure()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111, projection='3d')

    for index in [0, 1]:
        indices = np.where(config == index)[0]
        (xs, ys, zs) = zip(*points[indices])

        c = 'C%d' % index
        ax.scatter(xs, ys, zs, c=c, marker='o', s=800)

    (xs, ys, zs) = zip(*points)
    lim = max([abs(e) for e in xs + ys + zs])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    plt.show()
