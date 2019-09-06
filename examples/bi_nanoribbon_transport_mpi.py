import numpy as np
import tb
from negf.recursive_greens_functions import recursive_gf
from examples import data_bi_nanoribbon
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

    path_to_xyz_file = 'input_samples/bi_nanoribbon_058.xyz'

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

    h = tb.Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=0.0)
    h.initialize(radial_dep)
    period = data_bi_nanoribbon.lattice_constant * np.array([1.0, 0.0, 0.0])
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_coupling_hamiltonians()

    tr = np.zeros((energy.shape[0]))

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)

        g_trans, grd, grl, gru, gr_left = recursive_gf(E, [h_l], [h_0 + L + R], [h_r])
        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        tr = np.array(tr)

    return tr


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    energy = np.linspace(-0.5, 0.5, 100)

    tr = main(energy)

    ax = plt.axes()
    ax.plot(energy, tr)
    ax.set_ylabel(r'Transmission probability')
    ax.set_xlabel(r'Energy (eV)')
    plt.show()

