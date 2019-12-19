import numpy as np
import tb
import negf
import matplotlib.pyplot as plt
from tb import Hamiltonian, HamiltonianSp
from tb import Orbitals
from tb.plotting import plot_bs_split, plot_atom_positions


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
    band_gaps = []
    band_structures = []

    path = "input_samples/SiNW2.xyz"
    # path = "tb/third_party/si_slab.xyz"

    # hamiltonian = Hamiltonian(xyz=path, nn_distance=1.1, lead_l=[[0, 0, 1]], lead_r=[[0, 4, 3]], so_coupling=0.06)

    # left_lead = [0, 33, 35, 17, 16, 51, 25,  9, 53, 68,  1,  8, 24]
    # right_lead = [40, 66, 58, 47, 48, 71, 72, 73, 74, 65]

    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 2])

    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4,
                              sort_func=sorting)
    hamiltonian.initialize()

    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)

    num_points = 20
    kk = np.linspace(0, 0.57, num_points, endpoint=True)

    band_structure = []

    for jj in range(num_points):
        print(jj)
        vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)
    band_structures.append(band_structure)

    cba = band_structure.copy()
    vba = band_structure.copy()

    cba[cba < 0] = 1000
    vba[vba > 0] = -1000

    band_gap = np.min(cba) - np.max(vba)
    band_gaps.append(band_gap)

    if True:
        plot_bs_split(kk, vba, cba)

    print('-------------------------------------------------')

    hl, h0, hr = hamiltonian.get_hamiltonians()
    # hl1, h01, hr1, subblocks = hamiltonian.get_hamiltonians_block_tridiagonal()

    energy = np.linspace(2.1, 2.5, 30)
    ef1 = 0
    tempr = 10
    tr = np.zeros(energy.shape)
    dos = np.zeros(energy.shape)

    damp = 0.0005j

    for j, E in enumerate(energy):
        print(j)
        L, R = negf.surface_greens_function(E, hl, h0, hr, iterate=True, damp=damp)

        # s01, s02 = h01[0].shape
        # s11, s12 = h01[-1].shape

        # h01[0] = h01[0] + R[:s01, :s01]
        # h01[-1] = h01[-1] + L[-s11:, -s12:]

        g_trans, grd, grl, gru, gr_left = negf.recursive_gf(E, [hl], [h0+L+R], [hr], damp=damp)

        # h01[0] = h01[0] - R[:s01, :s01]
        # h01[-1] = h01[-1] - L[-s11:, -s12:]

        dos[j] = -2*np.trace(np.imag(grd[0]))

        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)

        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

    plt.plot(energy, dos)
    plt.show()

    plt.plot(energy, tr)
    plt.show()


if __name__ == '__main__':

    main()
