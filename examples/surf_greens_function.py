import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
from nanonet.negf.greens_functions import simple_iterative_greens_function, sancho_rubio_iterative_greens_function, surface_greens_function


def main(surf_greens_fun):
    """ An example for the Green's function usage"""

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)

    tb.Orbitals.orbital_sets = {'A': a}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5})

    xyz_file = """1
    A cell
    A1       0.0000000000    0.0000000000    0.0000000000
    """

    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    h.initialize()
    h.set_periodic_bc([[0, 0, 1.0]])
    h_l, h_0, h_r = h.get_hamiltonians()

    energy = np.linspace(-3.0, 1.5, 700)

    sgf_l = []
    sgf_r = []

    for E in energy:
        sf = surf_greens_fun(E, h_l, h_0, h_r , damp=0.001j)
        if isinstance(sf, tuple):
            L = sf[0]
            R = sf[1]
        else:
            L = sf
            R = surf_greens_fun(E, h_r, h_0, h_l , damp=0.001j)

        sgf_l.append(L)
        sgf_r.append(R)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j] = np.real(np.trace( np.linalg.multi_dot([ gamma_l, gf0, gamma_r, gf0.conj().T ]) ))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))

    fig, axs = plt.subplots(2, figsize=(5, 7))
    fig.suptitle('Green\'s function technique')
    axs[0].plot(energy, dos, 'k')
    # axs[0].title.set_text('Density of states')
    axs[0].set_xlabel('Energy (eV)')
    axs[0].set_ylabel('DOS')
    axs[1].plot(energy, tr, 'k')
    # axs[1].title.set_text('Transmission function')
    axs[1].set_xlabel('Energy (eV)')
    axs[1].set_ylabel('Transmission probability')
    plt.show()


def main1(surf_greens_fun):
    """ An example for the Green's function usage"""

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)
    b = tb.Orbitals('B')
    b.add_orbital('s', -0.5)
    c = tb.Orbitals('C')
    c.add_orbital('s', -0.3)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b, 'C': c}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
                     PARAMS_B_B={'ss_sigma': -0.5},
                     PARAMS_A_B={'ss_sigma': -0.5},
                     PARAMS_B_C={'ss_sigma': -0.5},
                     PARAMS_A_C={'ss_sigma': -0.5})

    xyz_file = """4
    H cell
    A1       0.0000000000    0.0000000000    0.0000000000
    B2       0.0000000000    0.0000000000    1.0000000000
    A2       0.0000000000    1.0000000000    0.0000000000
    B3       0.0000000000    1.0000000000    1.0000000000
    """

    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    h.initialize()
    h.set_periodic_bc([[0, 0, 2.0]])
    h_l, h_0, h_r = h.get_hamiltonians()

    energy = np.linspace(-3.0, 1.5, 700)

    sgf_l = []
    sgf_r = []

    for E in energy:
        sf = surf_greens_fun(E, h_l, h_0, h_r, damp=0.001j)
        if isinstance(sf, tuple):
            L = sf[0]
            R = sf[1]
        else:
            L = sf
            R = surf_greens_fun(E, h_r, h_0, h_l, damp=0.001j)

        sgf_l.append(L)
        sgf_r.append(R)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j]  = np.real(np.trace( np.linalg.multi_dot([ gamma_l, gf0, gamma_r, gf0.conj().T ]) ))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))

    fig, axs = plt.subplots(2, figsize=(5, 7))
    fig.suptitle('Green\'s function technique')
    axs[0].plot(energy, dos, 'k')
    # axs[0].title.set_text('Density of states')
    axs[0].set_xlabel('Energy (eV)')
    axs[0].set_ylabel('DOS')
    axs[1].plot(energy, tr, 'k')
    # axs[1].title.set_text('Transmission function')
    axs[1].set_xlabel('Energy (eV)')
    axs[1].set_ylabel('Transmission probability')
    plt.show()


if __name__ == "__main__":

    main(surf_greens_fun=surface_greens_function)
    main(surf_greens_fun=simple_iterative_greens_function)
    main(surf_greens_fun=sancho_rubio_iterative_greens_function)
    main1(surf_greens_fun=surface_greens_function)
    main1(surf_greens_fun=simple_iterative_greens_function)
    main1(surf_greens_fun=sancho_rubio_iterative_greens_function)
