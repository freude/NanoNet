"""
This example script computes band structure of the bulk silicon.
"""
import numpy as np
import matplotlib.pyplot as plt
from nanonet.tb import HamiltonianSym
from nanonet.tb import Orbitals
from sympy import pprint


# Lattice constant
a = 5.50

# Primitive cell
primitive_cell = a * np.array([[0.0, 0.5, 0.5],
                               [0.5, 0.0, 0.5],
                               [0.5, 0.5, 0.0]])

# High symmetry points
SPECIAL_K_POINTS_SI = {
    'GAMMA': [0.0, 0.0, 0.0],
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
    h = HamiltonianSym(xyz=path_to_xyz_file)
    h.initialize()

    # set periodic boundary conditions
    h.set_periodic_bc(primitive_cell)

    # compute band structure
    # ans = h.get_tb_matrix(np.array(SPECIAL_K_POINTS_SI['GAMMA']))
    ans = h.get_tb_element(0, 0, np.array(SPECIAL_K_POINTS_SI['GAMMA']))
    pprint(ans)


if __name__ == '__main__':

    main()
