"""Scripts to plot a band structure"""
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms


plt.style.use('plots.mplstyle')


def plot_bs_1D(k_points, band_structure, **kwargs):

    atoms = kwargs.get('atoms', None)
    title = kwargs.get('title', None)
    ylim = kwargs.get('ylim', None)

    ax = plt.axes()
    ax.set_title(title)
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, band_structure, 'k')
    ax.xaxis.grid()
    plt.ylim(ylim)
    plt.show()

    if atoms is not None:

        coords = atoms.get_positions()
        dim = len(np.nonzero(np.sum(coords, axis=0))[0])

        if dim==2:
            rotation = '90x,0y,00z'
        else:
            rotation = '10x,50y,30z'

        ax1 = plot_atoms(atoms, show_unit_cell=2, rotation=rotation)
        ax1.axis('off')
        plt.show()


def plot_bs_path(band_structure, **kwargs):

    size = band_structure

    num_points = kwargs.get('num_points', [size[0]])
    point_labels = kwargs.get('point_labels', None)
    title = kwargs.get('title', None)
    ylim = kwargs.get('ylim', None)

    plt.figure()
    ax = plt.axes()
    ax.set_title(title)
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(band_structure), 'k')

    if isinstance(point_labels, list) and len(point_labels) != 0:
        ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
        plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=point_labels)
    ax.xaxis.grid()
    plt.ylim(ylim)
    plt.show()


def plot_bs_2D(X, Y, band_structure, title):

    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.plot_surface(X, Y, band_structure[:, :, 2], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, band_structure[:, :, 3], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_zlabel('Energy (eV)')
    plt.show()
