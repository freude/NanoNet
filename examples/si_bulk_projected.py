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
    h = Hamiltonian(xyz=path_to_xyz_file)
    h.initialize()
    proj_s = np.matrix(np.diag([1, 0, 0, 0, 0, 0, 0, 0, 0, 1] * h.num_of_nodes))

    band_structure = []
    s_structure = []

    for ii, item in enumerate(k_points):
        h.set_periodic_bc(data_si_bulk.primitive_cell)
        [eigenvalues, vec] = h.diagonalize_periodic_bc(k_points[ii])
        band_structure.append(eigenvalues)

        measure_s = np.zeros(vec.shape[0], dtype=np.complex)

        for j, item in enumerate(vec.T):
            vector = np.matrix(item)
            measure_s[j] = vector * proj_s * vector.H / (vector * vector.H)

        s_structure.append(measure_s)

    band_structure = np.array(band_structure)
    s_structure = np.array(s_structure)

    import matplotlib as mpl

    for j, item in enumerate(band_structure.T):
        plt.scatter(np.arange(len(k_points[:, 0])), item, c=np.abs(s_structure[:, j]),
                    norm=mpl.colors.Normalize(vmin=0, vmax=.1))

    plt.show()

if __name__ == '__main__':

    main()
