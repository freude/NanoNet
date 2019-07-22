import numpy as np
import tb
from negf.recursive_greens_functions import recursive_gf
from examples import data_bi_nanoribbon
from tb.aux_functions import get_k_coords
import matplotlib.pyplot as plt


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

    path_to_xyz_file = 'input_samples/bi_nanoribbon.xyz'

    bi = tb.Orbitals('Bi')
    bi.add_orbital("s",  energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=0)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=0)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=0)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=0)
    bi.add_orbital("s",  energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=1)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=1)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=1)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=1)

    tb.Orbitals.orbital_sets = {'Bi': bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_nanoribbon.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_nanoribbon.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_nanoribbon.PARAMS_BI_BI3)

    sym_points = ['GAMMA', 'X']
    num_points = [50]
    k_points = get_k_coords(sym_points, num_points, data_bi_nanoribbon.SPECIAL_K_POINTS_BI_NANORIBBON)

    band_structure = []
    h = tb.Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=1.5)
    h.initialize(radial_dep)
    period = data_bi_nanoribbon.lattice_constant * np.array([1.0, 0.0, 0.0])
    h.set_periodic_bc([period])
    for jj, item in enumerate(k_points):
        [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
        band_structure.append(eigenvalues)
    band_structure = np.array(band_structure)

    ax = plt.axes()
    ax.plot(band_structure)
    plt.ylim((-1, 1))
    plt.show()


if __name__ == '__main__':

    main()

