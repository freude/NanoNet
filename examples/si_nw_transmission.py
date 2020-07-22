"""
This example script computes DOS and transmission function for the silicon nanowire
using the recursive Green's function algorithm.
"""
import logging
import numpy as np
import nanonet.negf as negf
import matplotlib.pyplot as plt
from nanonet.tb import Hamiltonian, HamiltonianSp
from nanonet.tb import Orbitals
from nanonet.tb.sorting_algorithms import sort_projection


def main():
    # use a predefined basis sets
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    # specify atomic coordinates file stored in xyz format
    path = "input_samples/SiNW2.xyz"

    # define leads indices
    right_lead = [0, 33, 35, 17, 16, 51, 25,  9, 53, 68,  1,  8, 24]
    left_lead = [40, 66, 58, 47, 48, 71, 72, 73, 74, 65, 6, 22, 7, 23, 14, 30, 15, 31]

    # create a Hamiltonian object storing the Hamiltonian matrices
    # hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4,
    #                           sort_func=sort_projection,
    #                           left_lead=left_lead, right_lead=right_lead)
    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4)
    hamiltonian.initialize()

    # set periodic boundary conditions
    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)

    # get Hamiltonian matrices
    hl, h0, hr = hamiltonian.get_hamiltonians()
    # get Hamiltonian matrices in the block-diagonal form
    hl1, h01, hr1, subblocks = hamiltonian.get_hamiltonians_block_tridiagonal()

    # specify energy array
    energy = np.linspace(2.1, 2.2, 50)

    # specify dephasing constant
    damp = 0.001j

    # initialize output arrays by zeros
    tr = np.zeros(energy.shape)
    dos = np.zeros(energy.shape)

    # energy loop
    for j, E in enumerate(energy):

        logging.info("{} Energy: {} eV".format(j, E))

        # compute self-energies describing boundary conditions at the leads contacts
        R, L = negf.surface_greens_function(E, hl, h0, hr, iterate=False, damp=damp)
        # R, L = surface_greens_function_poles_Shur(E, hl, h0, hr)
        # R, L = surface_greens_function(E, hl, h0, hr, iterate=False, damp=damp)

        s01, s02 = h01[0].shape
        s11, s12 = h01[-1].shape

        # apply self-energies
        # h01[0] = h01[0] + L[:s01, :s02]
        # h01[-1] = h01[-1] + R[-s11:, -s12:]

        # compute Green's functions using the recursive Green's function algorithm
        g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, [hl, hl], [h0+L, h0, h0+R], [hr, hr], damp=damp)
        # g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, hl1, h01, hr1, damp=damp)

        # detach self-energies
        # h01[0] = h01[0] - L[:s01, :s02]
        # h01[-1] = h01[-1] - R[-s11:, -s12:]

        # number of subblocks
        num_periods = len(grd)

        # compute DOS
        for jj in range(num_periods):
            # dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].conj().T))) / num_periods
            dos[j] = dos[j] - np.real(np.trace(np.imag(grd[jj]))) / num_periods

        # gamma_l = 1j * (L[:s01, :s02] - L[:s01, :s02].conj().T)
        # gamma_r = 1j * (R[-s11:, -s12:] - R[-s11:, -s12:].conj().T)
        gamma_l = 1j * (L - L.conj().T)
        gamma_r = 1j * (R - R.conj().T)

        # compute transmission spectrum
        tr[j] = np.real(np.trace(gamma_l.dot(g_trans).dot(gamma_r).dot(g_trans.conj().T)))

    # visualize
    plt.plot(energy, dos)
    plt.show()

    plt.plot(dos)
    plt.show()

    plt.plot(energy, tr)
    plt.show()


if __name__ == '__main__':

    main()
