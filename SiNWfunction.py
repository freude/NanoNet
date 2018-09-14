import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tb
import matplotlib.pyplot as plt



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

            widths = [s for s in xyz_file if s.isdigit()]
            width = int(''.join(widths))
            print(width)

            hamiltonian = tb.Hamiltonian(xyz=os.path.join(path, xyz_file), nn_distance=2.4)
            hamiltonian.initialize()

            if flag:
                plt.axis('off')
                plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
                plt.savefig('hamiltonian.pdf')
                plt.show()

            a_si = 5.50
            PRIMITIVE_CELL = [[0, 0, a_si]]
            hamiltonian.set_periodic_bc(PRIMITIVE_CELL)

            num_points = 20
            kk = np.linspace(0, 0.57, num_points, endpoint=True)

            band_structure = []

            for jj in xrange(num_points):
                print(jj)
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

            if flag:
                fig, ax = plt.subplots(1, 2)
                ax[0].set_ylim(-1.0, -0.3)
                ax[0].plot(kk, np.sort(np.real(vba)))
                ax[0].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
                ax[0].set_ylabel(r'Energy (eV)')
                ax[0].set_title('Valence band, NW width={} u.c.'.format(width))
                plt.savefig('bs_vb.pdf')

                ax[1].set_ylim(2.0, 2.7)
                ax[1].plot(kk, np.sort(np.real(cba)))
                ax[1].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
                ax[1].set_ylabel(r'Energy (eV)')
                ax[1].set_title('Conduction band, NW width={} u.c.'.format(width))
                fig.tight_layout()
                plt.savefig('bs_cb.pdf')

    return band_gaps, band_structures
