import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tb
import matplotlib.pyplot as plt

# Trying to make computations for the sequence of SiNW.xyz files
# Trying to write a script which looks for files with specific 
# names in a current directory and process them one by one
#os walk looks for xyz extension in directory 

import os

path = 'c:\users\sammy\desktop\NanoNet\input_samples'

tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

for xyz_file in os.listdir(path):
    if xyz_file.endswith('xyz'):
        hamiltonian = tb.Hamiltonian(xyz=os.path.join(path, xyz_file), nn_distance=2.4)
        hamiltonian.initialize()

        plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
        plt.savefig('hamitonian.pdf')

        a_si = 5.50
        PRIMITIVE_CELL = [[0, 0, 0.5 * a_si]]
        hamiltonian.set_periodic_bc(PRIMITIVE_CELL)
        k = np.linspace(0.0, 100, 100)
        k = np.vstack((np.zeros(k.shape), np.zeros(k.shape), k)).T

        vals = np.zeros((k.shape[0], hamiltonian.h_matrix.shape[0]), dtype=np.complex)

        for jj, i in enumerate(k):
            vals[jj, :], _ = hamiltonian.diagonalize_periodic_bc(list(i))

        import matplotlib.pyplot as plt
        plt.plot(np.sort(np.real(vals)))
        plt.show()

# Determining K-points where we want to compute band structure

num_points = 200
kk = np.linspace(0, 3.14/2, num_points, endpoint=True)

#Band structure computation for each K-point

band_structure = []

for jj in xrange(num_points):
    vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
    band_structure.append(vals)

band_structure = np.array(band_structure)

from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
ax = plt.axes()
ax.set_title('Band structure of Silicon Nanowire')
ax.set_xlabel(r'Wave vector (k)')
ax.set_ylabel(r'Energy (eV)')
ax.plot(kk, np.sort(np.real(band_structure)))
plt.show()