import sys
import matplotlib.pyplot as plt
import numpy as np
from nanonet.negf.hamiltonian_chain import HamiltonianChain
from nanonet.negf.recursive_greens_functions import recursive_gf
from nanonet.negf.field import Field1D
import nanonet.negf as negf
import nanonet.tb as tb


def complex_chain():
    """ """
    sys.path.insert(0, '/home/mk/TB_project/tb')

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)
    b = tb.Orbitals('B')
    b.add_orbital('s', -0.5)
    c = tb.Orbitals('C')
    c.add_orbital('s', -0.3)

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

    energy = np.linspace(-3.0, 1.5, 1700)

    sgf_l = []
    sgf_r = []

    for E in energy:
        left_se, right_se = negf.surface_greens_function(E, h_l, h_0, h_r, iterate=5, damp=0.05j)
        sgf_l.append(left_se)
        sgf_r.append(right_se)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    tr = np.zeros((energy.shape[0]))
    dos = np.zeros((energy.shape[0]))

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j] = np.real(np.trace(gamma_l.dot(gf0).dot(gamma_r).dot(gf0.conj().T)))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))

    return energy, dos, tr, h, sgf_l, sgf_r


def qw(coord, coords_of_steps, jumps, width=1):
    """

    Parameters
    ----------
    coord :

    coords_of_steps :

    jumps :

    width :
         (Default value = 1)

    Returns
    -------

    """
    ans = 0
    for j, item in enumerate(coords_of_steps):
        ans += jumps[j] * 0.5 * (np.tanh(coord - item) / width + 1.0)

    return ans


def z_dependence(coord):
    """

    Parameters
    ----------
    coord :


    Returns
    -------

    """
    coords_of_steps = [-14.0, -11.0, 11.0, 14.0]
    jumps = [-1.7, 1.7, -1.7, 1.7]

    return qw(coord, coords_of_steps, jumps)


f = Field1D(z_dependence, axis=2)

ef1 = 0.25 - 1.5
ef2 = 0.5 - 1.5
tempr = 10

energy, dos, tr, h, sgf_l, sgf_r = complex_chain()
h_l, h_0, h_r = h.get_hamiltonians()
cell = h.ct.pcv

h_chain = HamiltonianChain(h_l, h_0, h_r, h.get_site_coordinates())

periods = 20
h_chain.translate(cell[0], periods, periods)
h_chain.add_field(f)

num_periods = 2 * periods + 1

dos1 = np.zeros((energy.shape[0]))
tr = np.zeros((energy.shape[0]))
dens = np.zeros((energy.shape[0], num_periods))

for j, E in enumerate(energy):

    h_chain.add_self_energies(sgf_l[j, :, :], sgf_r[j, :, :], energy=E, tempr=tempr, ef1=ef1, ef2=ef2)
    g_trans, grd, grl, gru, gr_left, gnd, gnl, gnu, gn_left = recursive_gf(E,
                                                                           h_chain.h_l,
                                                                           h_chain.h_0,
                                                                           h_chain.h_r,
                                                                           s_in=h_chain.sgf,
                                                                           damp=0.0005j)
    h_chain.remove_self_energies()

    gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
    gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)

    tr[j] = np.real(np.trace(gamma_r.dot(g_trans).dot(gamma_l).dot(g_trans.conj().T)))

    for jj in range(num_periods):
        dos1[j] = dos1[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].conj().T))) / num_periods
        dens[j, jj] = 2 * np.trace(gnd[jj])


plt.figure(1)
plt.contourf(dens)

plt.figure(2)
plt.plot(dos1)

plt.show()