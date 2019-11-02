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


def main(energy, ef1, tempr):

    def fd(energy, ef, temp):
        kb = 8.61733e-5  # Boltzmann constant in eV
        return 1.0 / (1.0 + np.exp((energy - ef) / (kb * temp)))

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
    h_l, h_0, h_r = h.get_hamiltonians()

    tr = np.zeros((energy.shape[0]))
    dens = np.zeros((energy.shape[0], h.num_of_nodes))

    diags = np.zeros((energy.shape[0], h.num_of_nodes))

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)
        sgf_l = -2.0 * np.matrix(np.imag(L) * fd(E, ef1, tempr))

        g_trans, grd, grl, gru, gr_left,\
        gnd, gnl, gnu, gin_left = recursive_gf(E,
                                               [h_l],
                                               [h_0 + L + R],
                                               [h_r],
                                               s_in=[sgf_l])

        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        gn_diag = np.diag(grd[0])
        gn_diag = np.reshape(gn_diag, (h.num_of_nodes, -1))
        diags[j, :] = 2 * np.sum(gn_diag, axis=1)

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        tr = np.array(tr)
        # dens[j] = 2 * np.trace(gnd)

        ind = np.argsort(np.array(h._coords)[:, 1])
        # gn_diag = np.concatenate((np.diag(gnd[0])[ind], np.diag(gnd[1])[ind], np.diag(gnd[2])[ind]))
        gn_diag = np.diag(np.imag(grd[0]))
        gn_diag = np.reshape(gn_diag, (h.num_of_nodes, -1))
        dens[j, :] = 2 * np.sum(gn_diag, axis=1)

    return tr, dens


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    path_to_dat_file = 'examples/data/bi_nanoribbon_090_transmission.dat'

    energy = np.linspace(-0.5, 0.5, 150)

    ef1 = 0
    tempr = 0

    tr, dens = main(energy, ef1, tempr)

    ax = plt.axes()
    ax.plot(energy, tr)
    ax.set_ylabel(r'Transmission probability')
    ax.set_xlabel(r'Energy (eV)')
    plt.show()

    plt.figure(figsize=[5, 10])
    ax = plt.axes()
    ax.contourf(np.arange(14), energy, dens, 200, cmap='terrain')
    ax.set_ylabel(r'Energy (eV)')
    ax.set_xlabel(r'Site number')
    plt.show()

    data_to_write = np.c_[energy[:, None], tr]
    np.savetxt(path_to_dat_file, np.c_[data_to_write])

