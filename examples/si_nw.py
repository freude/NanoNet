import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
from tb.plotting import plot_bs_split, plot_atom_positions


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
    """
    This function computes the band gap / band structure for Silicon Nanowire
    :param path: directory path to where xyz input files are stored
    :param flag: boolean statements
    :return: band gap / band structure
    """
    # define orbitals sets
    Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    band_gaps = []
    band_structures = []

    path = "input_samples/SiNW2.xyz"

    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4)
    hamiltonian.initialize()

    if True:
        plt.axis('off')
        plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
        plt.savefig('hamiltonian.pdf')
        plt.show()

    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 20
    kk = np.linspace(0, 0.57, num_points, endpoint=True)

    band_structure = []

    for jj in range(num_points):
        print(jj)
        vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)
    band_structures.append(band_structure)

    cba = band_structure.copy()
    vba = band_structure.copy()

    cba[cba < 0] = 1000
    vba[vba > 0] = -1000

    band_gap = np.min(cba) - np.max(vba)
    band_gaps.append(band_gap)

    if True:
        plot_bs_split(kk, vba, cba)


if __name__ == '__main__':

    main()
