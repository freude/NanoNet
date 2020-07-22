"""
This example script computes band structure of the bulk silicon.
"""
import numpy as np
import matplotlib.pyplot as plt
from nanonet.tb import Hamiltonian
from nanonet.tb import Orbitals
from nanonet.tb import get_k_coords


# Lattice constant
a = 5.50

# Primitive cell
primitive_cell = a * np.array([[0.0, 0.5, 0.5],
                               [0.5, 0.0, 0.5],
                               [0.5, 0.5, 0.0]])

# High symmetry points
SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * np.pi / a, 0],
    'L': [np.pi / a,  np.pi / a,  np.pi / a],
    'W': [np.pi / a,  2 * np.pi / a,  0],
    'U': [np.pi / (2 * a), 2 * np.pi / a, np.pi/ (2 * a)],
    'K': [3 * np.pi / (2 * a), 3 * np.pi / (2 * a), 0]
}


def main():
    # specify atomic coordinates in xyz format
    path_to_xyz_file = """2
                          Bulk Si cell
                          Si1       0.000    0.000    0.000
                          Si2       1.375    1.375    1.375"""

    # specify basis set
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S'}

    # create a Hamiltonian object storing the Hamiltonian matrices
    h = Hamiltonian(xyz=path_to_xyz_file)
    h.initialize()

    # set periodic boundary conditions
    h.set_periodic_bc(primitive_cell)

    # define wave vector coordinates
    sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
    num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]
    k_points = get_k_coords(sym_points, num_points, 'Si')

    # compute band structure
    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    ax = plt.axes()
    ax.set_title('Band structure of the bulk silicon')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(np.real(band_structure))[:, :8], 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


if __name__ == '__main__':

    main()
