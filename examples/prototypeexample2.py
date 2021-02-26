import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
# noinspection PyUnresolvedReferences
from nanonet.negf.greens_functions import simple_iterative_greens_function, sancho_rubio_iterative_greens_function, \
    surface_greens_function

a = tb.Orbitals('A')
a.add_orbital('s', -1.0)

tb.Orbitals.orbital_sets = {'A': a}

tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5})

damp_value = 0.05j  # Large damp value 0.05 to provide noticeable smearing

xyz_file2 = """3
A cell
A1       0.0000000000    0.0000000000    0.0000000000
A2       1.0000000000    0.0000000000    0.0000000000
A3       2.0000000000    0.0000000000    0.0000000000
"""

h = tb.Hamiltonian(xyz=xyz_file2, nn_distance=1.1)
h.initialize()
h.set_periodic_bc([[0, 0, 1.0]])
h_l, h_0, h_r = h.get_hamiltonians()

energy = np.linspace(-4.5, 2.5, 716)

sgf_l = []
sgf_r = []

gam_l_lam = []  #We won't store Gamma, we'll store their decomposition.
gam_l_v = []
gam_r_lam = []
gam_r_v = []

for E in energy:
    # Note that though the surface Green's function technique is very fast, it can
    # have slight errors due to choice of numerical cutoffs, if the solution is
    # accurate the simple iterative will return the same answer immediately
    sf = surface_greens_function(E, h_l, h_0, h_r, damp=damp_value)
    L = sf[0]
    R = sf[1]

    L = simple_iterative_greens_function(E, h_l, h_0, h_r, damp=damp_value, initialguess=L)
    R = simple_iterative_greens_function(E, h_r, h_0, h_l, damp=damp_value, initialguess=R)

    laml, vl = np.linalg.eigh( 1j*( L - L.conj().T ) )
    lamr, vr = np.linalg.eigh( 1j*( R - R.conj().T ) )

    sgf_l.append(L)
    sgf_r.append(R)
    gam_l_lam.append(laml)
    gam_l_v.append(vl)
    gam_r_lam.append(lamr)
    gam_r_v.append(vr)


sgf_l = np.array(sgf_l)
sgf_r = np.array(sgf_r)
gam_l_lam = np.array(gam_l_lam)
gam_l_v = np.array(gam_l_v)
gam_r_lam = np.array(gam_r_lam)
gam_r_v = np.array(gam_r_v)

num_sites = h_0.shape[0]
gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)
dos = -np.trace(np.imag(gf), axis1=1, axis2=2) # should be anti-hermitian part, sloppy.
tr = np.zeros((energy.shape[0]), dtype=complex)

for j, E in enumerate(energy):
    gf0 = gf[j, :, :]
    gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
    gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
    tr[j] = np.real(np.trace(np.linalg.multi_dot([gamma_l, gf0, gamma_r, gf0.conj().T])))
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
plt.show(block=False)


fig, axs = plt.subplots(2, figsize=(5, 7))
fig.suptitle('Green\'s function technique')
axs[0].plot(energy, gam_l_lam, 'k') # np.sum(gam_l_lam, 1)
# axs[0].title.set_text('Density of states')
axs[0].set_xlabel('Energy (eV)')
axs[0].set_ylabel('Eigenvalues (left)')
axs[1].plot(energy, gam_r_lam, 'k')
# axs[1].title.set_text('Transmission function')
axs[1].set_xlabel('Energy (eV)')
axs[1].set_ylabel('Eigenvalues (right)')
plt.show(block=False)


plt.show()
