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


def make_tb_matrices():

    path_to_xyz_file = 'input_samples/bi_nanoribbon_014.xyz'

    bi = tb.Orbitals('Bi')
    bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=0)
    bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=0)
    bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic= 1, spin=0)
    bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic= 0, spin=0)
    bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic= 0, spin=1)
    bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=1)
    bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic= 1, spin=1)
    bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic= 0, spin=1)

    tb.Orbitals.orbital_sets = {'Bi': bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_nanoribbon.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_nanoribbon.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_nanoribbon.PARAMS_BI_BI3)

    h = tb.Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=2.0)
    h.initialize(radial_dep)
    period = data_bi_nanoribbon.lattice_constant * np.array([1.0, 0.0, 0.0])
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_hamiltonians()

    return h_l, h_0, h_r, h.num_of_nodes


def test_gf(energy, h_l, h_0, h_r, num_of_nodes):

    diags = np.zeros((energy.shape[0], num_of_nodes), dtype=np.complex)

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=2)

        g_trans, grd, grl, gru, gr_left = recursive_gf(E,
                                                       [h_l],
                                                       [h_0 + L + R],
                                                       [h_r])

        gn_diag = np.diag(grd[0])
        # gn_diag = np.reshape(gn_diag, (h._orbitals_dict['Bi'].num_of_orbitals, -1))
        gn_diag = np.reshape(gn_diag, (num_of_nodes, -1))
        diags[j, :] = 2 * np.sum(gn_diag, axis=1)

    return diags


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from negf.cfr import CFR

    h_r, h_0, h_l, num_of_nodes = make_tb_matrices()

    tempr = 300
    fd = CFR(40)
    Ef = np.linspace(-1.0, 1.0, 40)
    ans = []
    moment = 1j * CFR.val_inf * test_gf(np.array([1j * CFR.val_inf]), h_l, h_0, h_r, num_of_nodes)

    for ef in Ef:
        print(ef)
        points = fd.genetate_integration_points(ef, tempr)
        gf_vals = test_gf(points, h_l, h_0, h_r, num_of_nodes)
        ans.append(fd.integrate1(gf_vals, tempr, zero_moment=moment))

    plt.plot(np.squeeze(np.array(ans)))
    plt.show()
