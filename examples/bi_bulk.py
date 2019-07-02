import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Orbitals
from tb.aux_functions import get_k_coords
from examples import data_bi_bulk


def radial_dep(coords):

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

    path_to_xyz_file = 'input_samples/bi_bulk.xyz'

    path_to_dat_file = 'examples/data/bi_bulk_bands.dat'

    Atom.orbital_sets = {'Bi': 'Bismuth'}

    so_couplings = np.linspace(1.5, 1.5, 1)
    sym_points = ['K', 'GAMMA', 'T', 'W', 'L', 'LAMBDA']
    num_points = [10, 10, 10, 10, 10]

    k_points = get_k_coords(sym_points, num_points, data_bi_bulk.SPECIAL_K_POINTS_BI)

    band_structure = []
    for ii, item in enumerate(so_couplings):
        h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.6, so_coupling=item)
        h.initialize(radial_dep)
        h.set_periodic_bc(data_bi_bulk.primitive_cell)
        for jj, item in enumerate(k_points):
            [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
            band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    k_index = np.linspace(0, 1, np.size(k_points, axis=0))
    band_structure_data = np.c_[np.tile(so_couplings[:, None], [np.sum(num_points), 1]), np.tile(k_index[:, None], [len(so_couplings), 1]), band_structure]
    np.savetxt(path_to_dat_file, np.c_[band_structure_data])

    ax = plt.axes()
    ax.plot(band_structure)
    plt.ylim((-15, 5))
    plt.show()


if __name__ == '__main__':

    main()
