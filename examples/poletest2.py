import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
# noinspection PyUnresolvedReferences
from nanonet.negf.greens_functions import simple_iterative_greens_function, sancho_rubio_iterative_greens_function, \
    surface_greens_function
from nanonet.negf import pole_summation_method
from nanonet.negf.pole_summation_method import fermi_fun, fermi_deriv, fermi_deriv2


a = tb.Orbitals('A')
a.add_orbital('s', 0)

tb.Orbitals.orbital_sets = {'A': a}

tb.set_tb_params(PARAMS_A_A={'ss_sigma': -1})

xyz_file = """15
A cell
A1       0.0000000000    0.0000000000    0.0000000000
A2       1.0000000000    0.0000000000    0.0000000000
A3       2.0000000000    0.0000000000    0.0000000000
A4       3.0000000000    0.0000000000    0.0000000000
A5       4.0000000000    0.0000000000    0.0000000000
A6       5.0000000000    0.0000000000    0.0000000000
A7       6.0000000000    0.0000000000    0.0000000000
A8       7.0000000000    0.0000000000    0.0000000000
A9       8.0000000000    0.0000000000    0.0000000000
A10      9.0000000000    0.0000000000    0.0000000000
A11     10.0000000000    0.0000000000    0.0000000000
A12     11.0000000000    0.0000000000    0.0000000000
A13     12.0000000000    0.0000000000    0.0000000000
A14     13.0000000000    0.0000000000    0.0000000000
A15     14.0000000000    0.0000000000    0.0000000000
"""

h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
h.initialize()
h.set_periodic_bc([[0, 0, 1.0]])
h_l, h_0, h_r = h.get_hamiltonians()

muL = -3.92
muR = -3.90
muC = 0.5*(muL + muR)
# muL = -3.915
# muR = -3.905
kT = 0.010
reltol = 10**-6
p = np.ceil( -np.log(reltol) )

numE = round((muR - muL + 2*p*kT)/(0.05*kT)) + 1

energy = np.linspace(muL - p*kT, muR + p*kT, numE)

sgf_l = []
sgf_r = []

for E in energy:
    # Note that though the surface Green's function technique is very fast, it can
    # have slight errors due to choice of numerical cutoffs, if the solution is
    # accurate the simple iterative will return the same answer immediately
    sf = surface_greens_function(E, h_l, h_0, h_r, damp=0.0001j)
    L = sf[0]
    R = sf[1]

    L = simple_iterative_greens_function(E, h_l, h_0, h_r, damp=0.0001j, initialguess=L)
    R = simple_iterative_greens_function(E, h_r, h_0, h_l, damp=0.0001j, initialguess=R)

    sgf_l.append(L)
    sgf_r.append(R)


sgf_l = np.array(sgf_l)
sgf_r = np.array(sgf_r)


num_sites = h_0.shape[0]
gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)
val01 = np.zeros(h_0.shape[0])
val02 = np.zeros(h_0.shape[0])
val03 = np.zeros(h_0.shape[0])
val00 = np.zeros(h_0.shape[0])
val04 = np.zeros(h_0.shape[0])


for j, E in enumerate(energy):
    gf0 = gf[j, :, :]
    val01 = val01 + np.diag(gf0)*(fermi_fun(E, muR, kT) - fermi_fun(E, muL, kT))
    val02 = val02 + np.diag(gf0)*(fermi_fun(E, muR, kT) - fermi_fun(E, muC, kT))
    val03 = val03 + np.diag(gf0)*(fermi_fun(E, muC, kT) - fermi_fun(E, muL, kT))
    val00 = val00 + np.diag(gf0)*(fermi_deriv(E, muC, kT))  # Analytical derivative
    val04 = val04 + np.diag(gf0) * (fermi_deriv2(E, muC, kT))  # Analytical second derivative

# for j, E in enumerate(energy):
#     gf0 = gf[j, :, :]
#     gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
#     gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
#     tr[j] = np.real(np.trace(np.linalg.multi_dot([gamma_l, gf0, gamma_r, gf0.conj().T])))
#     dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))
#
# fig, axs = plt.subplots(2, figsize=(5, 7))
# fig.suptitle('Green\'s function technique')
# axs[0].plot(energy, dos, 'k')
# # axs[0].title.set_text('Density of states')
# axs[0].set_xlabel('Energy (eV)')
# axs[0].set_ylabel('DOS')
# axs[1].plot(energy, tr, 'k')
# # axs[1].title.set_text('Transmission function')
# axs[1].set_xlabel('Energy (eV)')
# axs[1].set_ylabel('Transmission probability')
# plt.show(block=False)


poles, residuesL, residuesR = pole_summation_method.pole_finite_difference(muL, muR, kT, reltol)


sgf2_l = []
sgf2_r = []

for E in poles:
    # Note that though the surface Green's function technique is very fast, it can
    # have slight errors due to choice of numerical cutoffs, if the solution is
    # accurate the simple iterative will return the same answer immediately
    sf = surface_greens_function(np.real(E), h_l, h_0, h_r, damp=1j*np.imag(E))
    L2 = sf[0]
    R2 = sf[1]

    L2 = simple_iterative_greens_function(np.real(E), h_l, h_0, h_r, damp=1j*np.imag(E), initialguess=L2)
    R2 = simple_iterative_greens_function(np.real(E), h_r, h_0, h_l, damp=1j*np.imag(E), initialguess=R2)

    sgf2_l.append(L2)
    sgf2_r.append(R2)


sgf2_l = np.array(sgf2_l)
sgf2_r = np.array(sgf2_r)


num_sites = h_0.shape[0]
gf2 = np.linalg.pinv(np.multiply.outer(poles, np.identity(num_sites)) - h_0 - sgf2_l - sgf2_r)

ldos = []

val1 = np.zeros(h_0.shape[0])
val2 = np.zeros(h_0.shape[0])
val3 = np.zeros(h_0.shape[0])
val4 = np.zeros(h_0.shape[0])


for j, E in enumerate(poles):
    gf00 = gf2[j, :, :]
#    ldostemp = -2*np.imag(np.diag(gf00))
#    ldos.append(ldostemp)
    val1 = val1 + residuesL[j]*np.diag(gf00)
    val2 = val2 + residuesR[j]*np.diag(gf00)
    val3 = val3 + 0.5*(residuesR[j] + residuesL[j])*np.diag(gf00)
    val4 = val4 + 0.5*(residuesR[j] - residuesL[j])*np.diag(gf00)


val00 = -2*np.imag(val00)*(energy[1]-energy[0])*.5
val01 = -2*np.imag(val01)/(muR-muL)*(energy[1]-energy[0])*.5
val02 = -2*np.imag(val02)/(muR-muC)*(energy[1]-energy[0])*.5
val03 = -2*np.imag(val03)/(muC-muL)*(energy[1]-energy[0])*.5
val04 = -2*np.imag(val04)*(energy[1]-energy[0])*.5/200  # Not sure where 200 is coming from, kT^2 ?
val1 = -2*np.imag(val1)/(muR-muL)
val2 = -2*np.imag(val2)/(muR-muL)
val3 = -2*np.imag(val3)/(muR-muL)
val4 = -2*np.imag(val4)/(muR-muL)

plt.plot(val00)
plt.show(block=False)

plt.plot(val01)
plt.show(block=False)

plt.plot(val02)
plt.show(block=False)

plt.plot(val03)
plt.show(block=False)

plt.plot(val04)
plt.show(block=False)

plt.plot(val1)
plt.show(block=False)

plt.plot(val2)
plt.show(block=False)

plt.plot(val3)
plt.show(block=False)

plt.plot(val4)
plt.show(block=False)

plt.show()
