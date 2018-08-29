import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tb
import matplotlib.pyplot as plt
import os

def bs(path='c:\users\sammy\desktop\NanoNet\input_samples', flag=True):
    """
    This function computes the band gap / band structure for Silicon Nanowire
    :param path: directory path to where xyz input files are stored
    :param flag: boolean statements
    :return: band gap / band structure
    """
    # define orbitals sets
    tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    band_gaps = []
    band_structures = []
    for xyz_file in os.listdir(path):
        if xyz_file.endswith('xyz'):
            hamiltonian = tb.Hamiltonian(xyz=os.path.join(path, xyz_file), nn_distance=2.4)
            hamiltonian.initialize()

        if flag:
            plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
            plt.savefig('hamitonian.pdf')
            plt.show()

            a_si = 5.50
            PRIMITIVE_CELL = [[0, 0, a_si]]
            hamiltonian.set_periodic_bc(PRIMITIVE_CELL)

            num_points = 50
            kk = np.linspace(0, 0.57, num_points, endpoint=True)
#k points
            band_structure = []

            for jj in xrange(num_points):
                vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
                band_structure.append(vals)

            band_structure = np.array(band_structure)
            band_structures.append(band_structure)

            cba = band_structure.copy()
            vba = band_structure.copy()

            cba[cba < 0] = 1000
            vba[vba > 0] = -1000

            band_gap = np.min(cba) - np.max(vba)
            band_gaps.append(band_gap)

            ax = plt.axes()
            ax.set_title('Band structure of Silicon Nanowire')
            ax.set_xlabel(r'Wave vector (k)')
            ax.set_ylabel(r'Energy (eV)')
            ax.plot(kk, np.sort(np.real(band_structure)))
        if flag:
            plt.show()

    return band_gaps, band_structures