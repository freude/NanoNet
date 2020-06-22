"""
This example demonstrates calculation of the band structure
for the silicon nanowire using a direct matrix diagonalization method.
"""
import numpy as np
from nanonet.tb import HamiltonianSp
from nanonet.tb import Orbitals
from nanonet.tb.plotting import plot_bs_split, plot_atom_positions


def main():
    # use a predefined basis sets
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    # specify atomic coordinates file in xyz format
    path = "input_samples/SiNW2.xyz"

    # create a Hamiltonian object in the sparse format
    hamiltonian = HamiltonianSp(xyz=path, nn_distance=2.4,)
    hamiltonian.initialize()

    # set periodic boundary conditions
    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(PRIMITIVE_CELL)

    # define wave vector coordinates
    num_points = 20
    kk = np.linspace(0, 0.57, num_points, endpoint=True)

    # compute band structure
    band_structure = []
    for jj in range(num_points):
        print("{}. Processing wave vector {}".format(jj, [0, 0, kk[jj]]))
        vals, _ = hamiltonian.diagonalize_periodic_bc([0, 0, kk[jj]])
        band_structure.append(vals)

    # visualize
    band_structure = np.array(band_structure)
    cba = band_structure.copy()
    vba = band_structure.copy()
    cba[cba < 0] = np.inf
    vba[vba > 0] = -np.inf
    plot_bs_split(kk, vba, cba)


if __name__ == '__main__':

    main()
