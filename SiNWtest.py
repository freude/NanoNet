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
        PRIMITIVE_CELL = [[0, 0, a_si]]
        hamiltonian.set_periodic_bc(PRIMITIVE_CELL)

        num_points = 200
        kk = np.linspace(0, 3.14 / 2, num_points, endpoint=True)

 # Band structure computation for each K-point

        band_structure = []

        for jj in xrange(num_points):
            vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
            band_structure.append(vals)

        band_structure = np.array(band_structure)

        ax = plt.axes()
        ax.set_title('Band structure of Silicon Nanowire')
        ax.set_xlabel(r'Wave vector (k)')
        ax.set_ylabel(r'Energy (eV)')
        ax.plot(kk, np.sort(np.real(band_structure)))
        plt.show()
