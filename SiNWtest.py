import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tb
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
