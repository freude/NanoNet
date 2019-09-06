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

    path_to_xyz_file = 'input_samples/bi_nanoribbon_014.xyz'

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
    num_points = [100]
    k_points = get_k_coords(sym_points, num_points, data_bi_nanoribbon.SPECIAL_K_POINTS_BI_NANORIBBON)

    band_structure = []
    spin_structure = []
    s_structure = []
    h = tb.Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=0.0)
    h.initialize(radial_dep)

    # --------------------------------------------------------------
    # proj_x = np.matrix(np.diag([1, 1, 1, 1, 0, 0, 0, 0] * h.num_of_nodes, 4) +\
    #          np.diag([1, 1, 1, 1] * h.num_of_nodes, -4))
    # proj_y = np.matrix(np.diag([-1j, -1j, -1j, -1j] * h.num_of_nodes, 4) + \
    #          np.diag([1j, 1j, 1j, 1j] * h.num_of_nodes, 4))
    proj_z = np.matrix(np.diag([1, 1, 1, 1, -1, -1, -1, -1] * h.num_of_nodes))
    proj_s = np.matrix(np.diag([1, 0, 0, 0, 1, 0, 0, 0] * h.num_of_nodes))

    # --------------------------------------------------------------

    period = data_bi_nanoribbon.lattice_constant * np.array([1.0, 0.0, 0.0])
    h.set_periodic_bc([period])
    for jj, item in enumerate(k_points):
        [eigenvalues, vec] = h.diagonalize_periodic_bc(k_points[jj])
        band_structure.append(eigenvalues)

        measure_spin = np.zeros(vec.shape[0], dtype=np.complex)
        measure_s = np.zeros(vec.shape[0], dtype=np.complex)

        for j, item in enumerate(vec.T):
            vector = np.matrix(item).T
            measure_spin[j] = vector.H * proj_z * vector
            measure_s[j] = vector.H * proj_s * vector

        spin_structure.append(measure_spin)
        s_structure.append(measure_s)

    band_structure = np.array(band_structure)
    spin_structure = np.array(spin_structure)
    s_structure = np.array(s_structure)

    import matplotlib as mpl
    fig, axs = plt.subplots(1, 2, figsize=(11, 7))

    spin_structure1 = np.copy(spin_structure)
    spin_structure2 = -np.copy(spin_structure)
    spin_structure1[spin_structure1 < 0] = 0.01
    spin_structure2[spin_structure2 < 0] = 0.01

    for j, item in enumerate(band_structure.T):
        # im1 = axs[0].scatter(k_points[:, 0], item, c=spin_structure[:, j],
        #                      norm=mpl.colors.Normalize(vmin=-1, vmax=1),
        #                      cmap='bwr')

        im1 = axs[0].scatter(k_points[:, 0], item, marker='o',
                             s=100*spin_structure1[:, j], facecolors='none', edgecolors='r')
        im1 = axs[0].scatter(k_points[:, 0], item, marker='v',
                             s=100*spin_structure2[:, j], facecolors='none', edgecolors='r')
    axs[0].set_title('Magnitude of the z-axis spin projection', fontsize=10)
    axs[0].set_ylim((-1, 1))
    # fig.colorbar(im1, ax=axs[0])

    for j, item in enumerate(band_structure.T):
        im2 = axs[1].scatter(k_points[:, 0], item, c=np.abs(s_structure[:, j]),
                             norm=mpl.colors.Normalize(vmin=0, vmax=.03))
    axs[1].set_title('s/p orbitals contribution', fontsize=10)
    axs[1].set_ylim((-1, 1))
    fig.colorbar(im2, ax=axs[1])
    plt.show()


if __name__ == '__main__':

    main()
