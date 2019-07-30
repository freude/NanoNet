import numpy as np
import tb
from negf.recursive_greens_functions import recursive_gf
from examples import data_bi_nanoribbon

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


def main(energy):

    def fd(energy, ef, temp):
        kb = 8.61733e-5  # Boltzmann constant in eV
        return 1.0 / (1.0 + np.exp((energy - ef) / (kb * temp)))

    ef1 = 0.45
    ef2 = -0.45
    tempr = 5

    path_to_xyz_file = 'input_samples/bi_nanoribbon_014.xyz'

    bi = tb.Orbitals('Bi')
    bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=0)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=0)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=0)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=0)
    bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=1)
    bi.add_orbital("px", energy= -0.486, principal=0, orbital=1, magnetic=-1, spin=1)
    bi.add_orbital("py", energy= -0.486, principal=0, orbital=1, magnetic= 1, spin=1)
    bi.add_orbital("pz", energy= -0.486, principal=0, orbital=1, magnetic= 0, spin=1)

    tb.Orbitals.orbital_sets = {'Bi': bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_nanoribbon.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_nanoribbon.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_nanoribbon.PARAMS_BI_BI3)

    h = tb.Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=2.0)
    h.initialize(radial_dep)
    period = data_bi_nanoribbon.lattice_constant * np.array([1.0, 0.0, 0.0])
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_coupling_hamiltonians()

    tr = np.zeros((energy.shape[0]))
    dens = np.zeros((energy.shape[0], h.num_of_nodes))

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)
        sgf_l = -2.0 * np.matrix(np.imag(L) * fd(E, ef1, tempr))
        sgf = np.matrix(np.zeros(h_0.shape))
        sgf_r = -2.0 * np.matrix(np.imag(L) * fd(E, ef2, tempr))

        g_trans, grd, grl, gru, gr_left,\
        gnd, gnl, gnu, gin_left = recursive_gf(E,
                                               [h_l],
                                               [h_0 + L + R],
                                               [h_r],
                                               s_in=[sgf_l - sgf_r])

        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        tr = np.array(tr)
        # dens[j] = 2 * np.trace(gnd)

        ind = np.argsort(np.array(h._coords)[:, 1])
        # gn_diag = np.concatenate((np.diag(gnd[0])[ind], np.diag(gnd[1])[ind], np.diag(gnd[2])[ind]))
        gn_diag = np.diag(gnd[0])
        gn_diag = np.reshape(gn_diag, (h._orbitals_dict['Bi'].num_of_orbitals, -1))
        dens[j, :] = 2 * np.sum(gn_diag, axis=0)

    return tr, dens


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    energy = np.linspace(-0.5, 0.5, 150)
    # energy = np.linspace(0.2, 0.4, 200)

    tr, dens = main(energy)

    ax = plt.axes()
    ax.plot(energy, tr)
    ax.set_ylabel(r'Transmission probability')
    ax.set_xlabel(r'Energy (eV)')
    plt.show()

