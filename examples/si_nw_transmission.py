"""
This example script computes DOS and transmission function for the silicon nanowire
using the recursive Green's function algorithm.
"""
import logging
import numpy as np
import nanonet.negf as negf
import matplotlib.pyplot as plt
from nanonet.tb import Hamiltonian
from nanonet.tb import Orbitals


def main():
    # use a predefined basis sets
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    # specify atomic coordinates file stored in xyz format
    path = "input_samples/SiNW2.xyz"

    # define leads indices
    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4)
    hamiltonian.initialize()

    # set periodic boundary conditions
    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)

    # get Hamiltonian matrices
    hl, h0, hr = hamiltonian.get_hamiltonians()

    # specify energy array
    energy = np.linspace(2.1, 2.5, 50)

    # specify dephasing constant
    damp = 0.001j

    # initialize output arrays by zeros
    tr = np.zeros(energy.shape)
    dos = np.zeros(energy.shape)

    # energy loop
    for j, E in enumerate(energy):

        logging.info("{0} Energy: {1:.4f} eV".format(j, E))

        # compute self-energies describing boundary conditions at the leads contacts
        L, R = negf.surface_greens_function(E, hl, h0, hr, iterate=True, damp=damp)

        # compute Green's functions using the recursive Green's function algorithm
        # g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, [hl, hl], [h0 + L, h0, h0 + R], [hr, hr], damp=damp)
        g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, [hl], [h0 + L + R], [hr], damp=damp)

        # number of subblocks
        num_blocks = len(grd)

        # compute DOS
        for jj in range(num_blocks):
            dos[j] = dos[j] - np.trace(np.imag(grd[jj])) / num_blocks

        # coupling matrices
        gamma_l = 1j * (L - L.conj().T)
        gamma_r = 1j * (R - R.conj().T)

        # compute transmission spectrum
        tr[j] = np.real(np.trace(gamma_l.dot(g_trans).dot(gamma_r).dot(g_trans.conj().T)))

    # visualize
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(energy, dos, 'k')
    ax[0].set_ylabel(r'DOS (a.u)')
    ax[0].set_xlabel(r'Energy (eV)')

    ax[1].plot(energy, tr, 'k')
    ax[1].set_ylabel(r'Transmission (a.u.)')
    ax[1].set_xlabel(r'Energy (eV)')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    main()
