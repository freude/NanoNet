"""
This example script computes band structure of the crystalline bismuth.
It uses the third-nearest neighbor approximation with a step-wise distance distance
associating distances with sets of TB parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from nanonet.tb import Hamiltonian
from nanonet.tb import Orbitals, set_tb_params
from nanonet.tb.aux_functions import get_k_coords
from examples.data_bi_bulk import SPECIAL_K_POINTS_BI, primitive_cell


def radial_dep(coords):
    """
        Step-wise radial dependence function
    """
    norm_of_coords = np.linalg.norm(coords)
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 3
    else:
        return 100


def main():
    # define atomic coordinates
    path_to_xyz_file = """2
                          Bi2 cell
                          Bi1       0.0    0.0    0.0
                          Bi2       0.0    0.0    5.52321494"""

    # define basis set
    bi_orb = Orbitals('Bi')
    bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=0)
    bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=0)
    bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=0)
    bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=0)
    bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=1)
    bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=1)
    bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=1)
    bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=1)

    # define TB parameters
    # 1NN - Bi-Bi
    PAR1 = {'ss_sigma': -0.608,
            'sp_sigma': 1.320,
            'pp_sigma': 1.854,
            'pp_pi': -0.600}

    # 2NN - Bi-Bi
    PAR2 = {'ss_sigma': -0.384,
            'sp_sigma': 0.433,
            'pp_sigma': 1.396,
            'pp_pi': -0.344}

    # 3NN - Bi-Bi
    PAR3 = {'ss_sigma': 0,
            'sp_sigma': 0,
            'pp_sigma': 0.156,
            'pp_pi': 0}

    set_tb_params(PARAMS_BI_BI1=PAR1, PARAMS_BI_BI2=PAR2, PARAMS_BI_BI3=PAR3)

    # compute Hamiltonian matrices
    h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.6, so_coupling=1.5)
    h.initialize(radial_dep)
    h.set_periodic_bc(primitive_cell)

    # define wave vectors
    sym_points = ['K', 'GAMMA', 'T', 'W', 'L', 'LAMBDA']
    num_points = [10, 10, 10, 10, 10]
    k_points = get_k_coords(sym_points, num_points, SPECIAL_K_POINTS_BI)

    # compute band structure
    band_structure = []

    for jj, item in enumerate(k_points):
        [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
        band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    # visualization
    ax = plt.axes()
    plt.ylim((-15, 5))
    ax.set_title('Band structure of the bulk bismuth')
    ax.set_ylabel('Energy (eV)')
    ax.plot(band_structure, 'k')
    ax.plot([0, len(band_structure)], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


if __name__ == '__main__':

    main()
