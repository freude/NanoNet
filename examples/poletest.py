import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb
# noinspection PyUnresolvedReferences
from nanonet.negf import pole_summation_method

muL = -1.0
muR = 1.0
kT = .25
reltol = 10**-12

poles, residuesL, residuesR = pole_summation_method.pole_finite_difference(muL, muR, kT, reltol)

plt.scatter(np.real(poles), np.imag(poles))
plt.title('Pole locations')
plt.xlabel('Re(Energy)')
plt.ylabel('Im(Energy)')
plt.show(block=False)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(np.real(poles), np.imag(poles), np.abs(residuesL)/kT)
ax.set_xlabel('Re(Energy)')
ax.set_ylabel('Im(Energy)')
ax.set_zlabel('abs(ResidueL)')
plt.show(block=False)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(np.real(poles), np.imag(poles), np.abs(residuesR)/kT)
ax.set_xlabel('Re(Energy)')
ax.set_ylabel('Im(Energy)')
ax.set_zlabel('abs(ResidueR)')
plt.show(block=False)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(np.real(poles), np.imag(poles), np.abs(residuesR+residuesL)/kT)
ax.set_xlabel('Re(Energy)')
ax.set_ylabel('Im(Energy)')
ax.set_zlabel('abs(ResidueL + ResidueR)')
plt.show(block=False)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(np.real(poles), np.imag(poles), np.abs(residuesR-residuesL)/(2*kT))
ax.set_xlabel('Re(Energy)')
ax.set_ylabel('Im(Energy)')
ax.set_zlabel('abs(ResidueR-ResidueL)/2')
plt.show(block=False)


plt.show()


