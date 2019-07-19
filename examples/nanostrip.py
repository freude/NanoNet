import numpy as np
import tb
from negf.recursive_greens_functions import recursive_gf


def main(energy):

    # ---------------------------------------------------------------------------------
    # ----------- compute tight-binding matrices and define energy scale --------------
    # ---------------------------------------------------------------------------------

    orb = tb.Orbitals('A')
    orb.add_orbital('s', energy=-1.0)

    tb.Orbitals.orbital_sets = {'A': orb}

    tb.set_tb_params(PARAMS_A_A={"ss_sigma": 1.0})

    input_file = """4
    Nanostrip
    A1 0.0 0.0 0.0
    A2 0.0 1.0 0.0
    A3 0.0 2.0 0.0
    A4 0.0 3.0 0.0
    
    """

    h = tb.Hamiltonian(xyz=input_file, nn_distance=1.4)
    h.initialize()
    period = [0, 0, 1.0]
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_coupling_hamiltonians()

    # ---------------------------------------------------------------------------------
    # -------------------- compute Green's functions of the system --------------------
    # ---------------------------------------------------------------------------------


    dos = np.zeros((energy.shape[0]))
    tr = np.zeros((energy.shape[0]))
    dens = np.zeros((energy.shape[0], 1))

    par_data = []

    ef1 = 2.1
    ef2 = 2.1
    tempr = 100

    for j, E in enumerate(energy):

        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)

        g_trans, grd, grl, gru, gr_left = recursive_gf(E, [h_l], [h_0 + L + R], [h_r])
        dos[j] = np.real(np.trace(1j * (grd[0] - grd[0].H)))
        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        tr = np.array(tr)

        dos = np.array(dos)

    return dos, tr


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    energy = np.linspace(-5.0, 5.0, 150)

    dos, tr = main(energy)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(energy, dos)
    ax[0].set_ylabel(r'DOS (a.u)')
    ax[0].set_xlabel(r'Energy (eV)')

    ax[1].plot(energy, tr)
    ax[1].set_ylabel(r'Transmission (a.u.)')
    ax[1].set_xlabel(r'Energy (eV)')
    fig.tight_layout()
    plt.show()