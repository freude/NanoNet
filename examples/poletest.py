import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
# noinspection PyUnresolvedReferences
from nanonet.negf import pole_summation_method

muL = -1.1
muR = 1.1
kT = 0.1
reltol = 10**-12

poles, residuesL, residuesR = pole_summation_method.pole_finite_difference(muL, muR, kT, reltol)

plt.scatter(np.real(poles), np.imag(poles))
plt.xlabel('Re(Energy)')
plt.ylabel('Im(Energy)')
plt.show(block=False)

plt.show()


