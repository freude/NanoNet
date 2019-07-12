import sys
import tb
sys.path.append('./negf')
sys.path.append('../negf')
import numpy as np
import matplotlib.pyplot as plt
from negf.hamiltonian_chain import HamiltonianChain
from negf.field import Field
from negf.recursive_greens_functions import recursive_gf


def compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True):
    sgf_l = []
    sgf_r = []
    factor = []
    factor1 = []
    factor2 = []

    num_sites = h_0.shape[0]

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=3)

        test_gf = E * np.identity(num_sites) - h_0 - L - R
        metrics = np.linalg.cond(test_gf)
        print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))

        # if metrics > 15000:
        #     R = iterate_gf(E, h_0, h_l, h_r, R, 1)
        #     L = iterate_gf(E, h_0, h_l, h_r, L, 1)

        sgf_l.append(L)
        sgf_r.append(R)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    if save:
        np.save('sgf_l', sgf_l)
        np.save('sgf_r', sgf_r)

    return sgf_l, sgf_r


h_l = np.matrix([[0.0, 1.0],
                      [0.0, 0.0]])

h_0 = np.matrix([[3.0, 1.0],
                      [1.0, 2.0]])

h_r = np.matrix([[0.0, 0.0],
                      [1.0, 0.0]])

coords = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.25]])

num_sites = h_0.shape[0]

# field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
field = Field(path='/home/mk/gpaw_swarm/gpaw_comp/tetracene_cation.cube')
field.set_origin(np.array([0.0, 5.0, 0.0]))

h = HamiltonianChain(h_l, h_0, h_r, coords)

period = np.array([0.0, 0.0, 0.5])
h.translate(period, 20, 20)
h.add_field(field, eps=1)

plt.imshow(h.get_matrix())
plt.show()

h_d = h.get_matrix()

energy = np.linspace(0, 5, 150)

sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=False)

mat_l_list, mat_d_list, mat_u_list = h.h_l, h.h_0, h.h_r

# energy = energy[5:25]

tr = np.zeros((energy.shape[0]), dtype=np.complex)
dos = np.zeros((energy.shape[0]), dtype=np.complex)

for j, E in enumerate(energy):
    print(j)

    mat_d_list[0] = mat_d_list[0] + sgf_r[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] + sgf_l[j, :, :]

    g_trans, grd, grl, gru, gr_left = recursive_gf(E, mat_l_list, mat_d_list, mat_u_list)

    mat_d_list[0] = mat_d_list[0] - sgf_r[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] - sgf_l[j, :, :]

    for jj in range(len(grd)):
        dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].H)))

    gamma_l = 1j * (np.matrix(sgf_r[j, :, :]) - np.matrix(sgf_r[j, :, :]).H)
    gamma_r = 1j * (np.matrix(sgf_l[j, :, :]) - np.matrix(sgf_l[j, :, :]).H)
    tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))


ax = plt.axes()
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('DOS')
ax.plot(energy, dos)
plt.show()

ax = plt.axes()
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Transmission coefficient (a.u.)')
ax.plot(energy, tr)
plt.show()
