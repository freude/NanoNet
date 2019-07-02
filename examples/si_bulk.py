import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Orbitals
from examples import data_si_bulk
from tb import get_k_coords



def radial_dep(coords):

    norm_of_coords = np.linalg.norm(coords)
    print(norm_of_coords)
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 3
    else:
        return 100


def main():

    path_to_xyz_file = 'input_samples/bulk_silicon.xyz'

    path_to_dat_file = 'examples/data/si_bulk_bands.dat'

    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S'}

    sym_points = ['L', 'GAMMA', 'X']
    num_points = [20, 20]

    k_points = get_k_coords(sym_points, num_points, data_si_bulk.SPECIAL_K_POINTS_SI)

    band_structure = []
    for ii, item in enumerate(k_points):
        h = Hamiltonian(xyz=path_to_xyz_file)
        h.initialize()
        h.set_periodic_bc(data_si_bulk.primitive_cell)
        [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[ii])
        band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    k_index = np.linspace(0, 1, np.size(k_points, axis=0))
    band_structure_data = np.c_[k_index[:, None], band_structure]
    np.savetxt(path_to_dat_file, np.c_[band_structure_data])

    ax = plt.axes()
    ax.plot(band_structure)
    plt.ylim((-15, 15))
    plt.show()


if __name__ == '__main__':

    main()
