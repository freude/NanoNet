import logging
import time
import numpy as np
import nanonet.negf as negf
import matplotlib.pyplot as plt
from nanonet.tb import Hamiltonian, HamiltonianSp
from nanonet.tb import Orbitals
from nanonet.tb.plotting import plot_bs_split, plot_atom_positions


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
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    path = "input_samples/SiNW2.xyz"

    # hamiltonian = Hamiltonian(xyz=path, nn_distance=1.1, lead_l=[[0, 0, 1]], lead_r=[[0, 4, 3]], so_coupling=0.06)

    right_lead = [0, 33, 35, 17, 16, 51, 25,  9, 53, 68,  1,  8, 24]
    left_lead = [40, 66, 58, 47, 48, 71, 72, 73, 74, 65]

    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 2])

    from nanonet.tb.sorting_algorithms import sort_projection

    start = time.time()
    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4, sort_func=sort_projection, left_lead=left_lead, right_lead=right_lead)
    hamiltonian.initialize()

    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)

    forming_time = time.time() - start
    print(forming_time)

    hl, h0, hr = hamiltonian.get_hamiltonians()
    hl1, h01, hr1, subblocks = hamiltonian.get_hamiltonians_block_tridiagonal()

    energy = np.linspace(2.1, 2.5, 50)
    tr = np.zeros(energy.shape)
    dos = np.zeros(energy.shape)

    damp = 0.0005j

    for j, E in enumerate(energy):
        logging.info("{} Energy: {} eV".format(j, E))
        start = time.time()
        R, L = negf.surface_greens_function(E, hl, h0, hr, iterate=True, damp=damp)
        sgf_time = time.time() - start
        print(sgf_time)
        s01, s02 = h01[0].shape
        s11, s12 = h01[-1].shape

        h01[0] = h01[0] + L[:s01, :s02]
        h01[-1] = h01[-1] + R[-s11:, -s12:]

        start = time.time()
        # g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, [hl, hl], [h0+L, h0, h0+R], [hr, hr], damp=damp)
        g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, hl1, h01, hr1, damp=damp)
        rgf_time = time.time() - start
        print(rgf_time)

        h01[0] = h01[0] - L[:s01, :s02]
        h01[-1] = h01[-1] - R[-s11:, -s12:]

        num_periods = 3

        for jj in range(num_periods):
            dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].conj().T))) / num_periods

        # dos[j] = -2 * np.trace(np.imag(grd[0]))

        gamma_l = 1j * (L[:s01, :s02] - L[:s01, :s02].conj().T)
        gamma_r = 1j * (R[-s11:, -s12:] - R[-s11:, -s12:].conj().T)

        tr[j] = np.real(np.trace(gamma_l.dot(g_trans).dot(gamma_r).dot(g_trans.conj().T)))

    plt.plot(energy, dos)
    plt.show()

    plt.plot(energy, tr)
    plt.show()


if __name__ == '__main__':

    main()
