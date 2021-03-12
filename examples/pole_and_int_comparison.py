'''
This example script computes the finite difference density
derivative using both numerical integration and complex pole
summation. It serves as a demonstration and a validation of
the pole summation method. For the given parameters below,
the numerical energy grid contains 348 energy points (it would
be larger if the lower bound wasn't truncated by Emin) and the
complex pole summation contains only 21. Not only do these
results agree within 1 part in 10,000 at 1/16th the compu-
tational cost, due to its construction, the complex pole sum-
mation is more accurate. Where possible, the complex pole
summation method should be used over numerical integration.
'''

import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
from nanonet.negf.greens_functions import simple_iterative_greens_function, surface_greens_function
from nanonet.negf import pole_summation_method
from nanonet.negf.pole_summation_method import fermi_fun

# First we design a tight-binding model. We choose a 15 site model
# so that it is symmetric and that features may be clearly ob-
# served, one or two site models would look to similar to clearly
# differentiate whether something erroneously similar was happening

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

# Now that the Hamiltonian is constructed within the TB
# framework, we set our numerical parameters to be
# examined. We choose two endpoints mu +/- dmu,
# the temperature being evaluated, the relative tolerance
# of the pole summation and the numerical integration

# Emin = -3.98, is the point where there are
# no more states to the left, we can use this
# to reduce number of points in the evaluation

Emin = -3.98
muL = -3.9175
muR = -3.9025
muC = 0.5*(muL + muR)  # This is the energy the derivative is being evaluated at
kT = 0.010
reltol = 10**-8
p = np.ceil(-np.log(reltol))  # Integer number of kT away to get the desired relative tolerance.

lowbnd = max(Emin, muL - p*kT)
uppbnd = muR + p*kT
# We chose to have our energy spacing be at most 3*kT/40
numE = round((uppbnd-lowbnd)/(0.075*kT)) + 1
# numE = round((muR - muL + 2*p*kT)/(0.075*kT)) + 1

# We generate our grid for numerical integration, paying mind about the FD tails at muL-p*kT and muR + p*kT.
energy = np.linspace(lowbnd, uppbnd, numE)
# energy = np.linspace(muL - p*kT, muR + p*kT, numE)

# Initialize the storage of the surface Green's funs.
sgf_l = []
sgf_r = []

# We compute the numerical evaluation of the Green's
# function on the real energy grid. This is by far
# the most expensive part of this example.

for E in energy:
    # Note that though the surface Green's function technique is very fast, it can
    # have slight errors due to choice of numerical cutoffs, if the solution is
    # accurate the simple iterative will return the same answer immediately
    # the user can comment it out if they wish to trust those evaluations.
    sf = surface_greens_function(E, h_l, h_0, h_r, damp=0.00001j)
    L = sf[0]
    R = sf[1]

    L = simple_iterative_greens_function(E, h_l, h_0, h_r, damp=0.00001j, initialguess=L)
    R = simple_iterative_greens_function(E, h_r, h_0, h_l, damp=0.00001j, initialguess=R)

    sgf_l.append(L)
    sgf_r.append(R)

# Convert the lists into arrays
sgf_l = np.array(sgf_l)
sgf_r = np.array(sgf_r)

# Compute the energy-wise Green's function
num_sites = h_0.shape[0]
gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

# We preallocate the arrays to store our integrated solutions
backwardsint = np.zeros(num_sites)
forwardsint = np.zeros(num_sites)
centralint = np.zeros(num_sites)
LDOS = []

# The integral is just a sum of the weighted diagonals for whichever integral we are computing
for j, E in enumerate(energy):
    gf0 = gf[j, :, :]
    gdiag = np.diag(gf0)
    backwardsint = backwardsint + gdiag*(fermi_fun(E, muC, kT) - fermi_fun(E, muL, kT))
    forwardsint = forwardsint + gdiag*(fermi_fun(E, muR, kT) - fermi_fun(E, muC, kT))
    centralint = centralint + gdiag*(fermi_fun(E, muR, kT) - fermi_fun(E, muL, kT))
    LDOS.append(np.sum(gdiag))

# Now we have completed our numerical integration,
# we move onto pole summation, simply choose the
# chemical potentials the temperature and the rel-
# ative error as before and the method spits out
# one set of poles and two sets of residues for
# the backwards and forwards differences with the
# central being their sum.
poles, residuesL, residuesR = pole_summation_method.pole_finite_difference(muL, muR, kT, reltol)

# Allocate the surface green's functions for poles.
sgf2_l = []
sgf2_r = []

# Do each of the poles, note that the imaginary
# part of the energy is the damping parameter
# and the real part works as normal.
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

gf2 = np.linalg.pinv(np.multiply.outer(poles, np.identity(num_sites)) - h_0 - sgf2_l - sgf2_r)

backwardspole = np.zeros(num_sites)
forwardspole = np.zeros(num_sites)
centralpole = np.zeros(num_sites)

# Add up the weighted contributions
for j, E in enumerate(poles):
    gf00 = gf2[j, :, :]
    gdiag2 = np.diag(gf00)
    backwardspole = backwardspole + residuesL[j]*gdiag2
    forwardspole = forwardspole + residuesR[j]*gdiag2
    centralpole = centralpole + (residuesR[j] + residuesL[j])*gdiag2

# dE to correct numerical integration
dE = (energy[1]-energy[0])
dmu = (muR-muL)/2

# Backwards Finite Diff First Derivative
backwardsint = -2*np.imag(backwardsint)*dE/(1*dmu)
backwardspole = -2*np.imag(backwardspole)/(1*dmu)

# Forwards Finite Diff First Derivative
forwardsint = -2*np.imag(forwardsint)*dE/(1*dmu)
forwardspole = -2*np.imag(forwardspole)/(1*dmu)

# Centred Finite Diff First Derivative
centralint = -2*np.imag(centralint)*dE/(2*dmu)
centralpole = -2*np.imag(centralpole)/(2*dmu)


# LDOS
LDOS = -2*np.imag(LDOS)
#plt.plot(energy,LDOS, color='#464646', linestyle='solid')
#plt.show()

# Works
plt.plot(backwardsint, color='#951158', linestyle='dashed')
plt.plot(backwardspole, color='#D40F7D')
plt.plot(forwardspole, color='#00808B', linestyle='dashed')
plt.plot(forwardsint, color='#00AEC7')
plt.plot(centralint, color='#899600', linestyle='dashed')
plt.plot(centralpole, color='#C4D600')

plt.show()

# Now we compute the relative difference between the two answers, note this isn't really an error
# as we're computing the same approximate thing two different ways, it's to check if they agree.
errbackward = np.log10(0.5*np.linalg.norm(backwardsint-backwardspole)/np.linalg.norm(backwardsint+backwardspole))
errforward = np.log10(0.5*np.linalg.norm(forwardsint-forwardspole)/np.linalg.norm(forwardsint+forwardspole))
errcentred = np.log10(0.5*np.linalg.norm(centralint-centralpole)/np.linalg.norm(centralint+centralpole))

print("Relative backwards differences discrepancy = 10^"+f'{errbackward:.3f}')
print("Relative forward differences discrepancy = 10^"+f'{errforward:.3f}')
print("Relative central differences discrepancy = 10^"+f'{errcentred:.3f}')
