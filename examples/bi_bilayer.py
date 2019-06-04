import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
import tb
from tb.aux_functions import get_k_coords
from examples import data_bi_bilayer


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

    path_to_xyz_file = 'input_samples/bi_bilayer.xyz'

    path_to_dat_file = 'examples/data/bi_bilayer_bands.dat'

    bi = Atom('Bi')
    bi.add_orbital("s",  energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=0)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=0)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=0)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=0)
    bi.add_orbital("s",  energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=1)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=1)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=1)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=1)

    Atom.orbital_sets = {'Bi': bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_bilayer.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_bilayer.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_bilayer.PARAMS_BI_BI3)

    so_couplings = np.linspace(1.5, 1.5, 1)
    sym_points = ['M', 'GAMMA', 'K', 'M']
    num_points = [10, 10, 10]

    k_points = get_k_coords(sym_points, num_points, data_bi_bilayer.SPECIAL_K_POINTS_BI)

    band_structure = []
    for ii, item in enumerate(so_couplings):
        h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=item)
        h.initialize(radial_dep)
        h.set_periodic_bc(data_bi_bilayer.primitive_cell)
        for jj, item in enumerate(k_points):
            [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
            band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    k_index = np.linspace(0, 1, np.size(k_points, axis=0))
    band_structure_data = np.c_[np.tile(so_couplings[:, None], [np.sum(num_points), 1]), np.tile(k_index[:, None], [len(so_couplings), 1]), band_structure]
    np.savetxt(path_to_dat_file, np.c_[band_structure_data])

    ax = plt.axes()
    ax.plot(band_structure)
    plt.ylim((-1, 1))
    plt.show()


if __name__ == '__main__':

    main()
