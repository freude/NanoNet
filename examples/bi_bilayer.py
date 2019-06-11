import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Orbitals
from examples import data_bi_bilayer
from tb.plotting import plot_atom_positions, plot_atom_positions1
import tb


bi = Orbitals('Bi')
bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=0)
bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=0)
bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=0)
bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=0)
bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=1)
bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=1)
bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=1)
bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=1)


# def radial_dep(coords):
#
#     norm_of_coords = np.linalg.norm(coords)
#
#     if norm_of_coords < 3.3:
#         return 1
#     elif 3.7 > norm_of_coords > 3.3:
#         return 2
#     elif 4.7 > norm_of_coords > 3.7:
#         return 100
#     else:
#         return 100

def radial_dep(coords):

    norm_of_coords = np.linalg.norm(coords)
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 3
    else:
        return 100


def main():

    from tb.aux_functions import get_k_coords

    path_to_xyz_file = """2
                              Bilayer Bismuth
                              Bi   -2.2666 -1.30862212 -1.59098161
                              Bi   0.0    0.0    0.0
                           """

    # path_to_pdf_file = '../band_structure_of_bulk_bismuth.pdf'
    path_to_data_file = '/Users/tcqp/Dropbox/research/conferences/2018/fleet/poster/data/band_structure_of_111_bilayer/etb/3.0/band_structure.csv'
    species = 'Bi'
    basis_set = 'Bismuth'
    sym_points = ['M', 'GAMMA', 'K', 'M']
    # sym_points = ['GAMMA', 'M', 'K', 'GAMMA']
    # sym_points = ['M', 'GAMMA', 'K']
    # sym_points = ['GAMMA', 'GAMMA']

    num_points = [40, 40, 40]
    # num_points = [1]
    indices_of_bands = range( 0, 16 )

    primitive_cell = data_bi_bilayer.cell

    Orbitals.orbital_sets = {species: bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_bilayer.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_bilayer.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_bilayer.PARAMS_BI_BI3)

    k_points = get_k_coords( sym_points, num_points, data_bi_bilayer.SPECIAL_K_POINTS_BI)

    list_of_spin_orbit_couplings = [3.0]
    # list_of_spin_orbit_couplings = np.linspace(0, 3.25, 160)

    band_structure = []
    for ii, item in enumerate(list_of_spin_orbit_couplings):

        h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=item)
        h.initialize(radial_dep)
        h.set_periodic_bc(primitive_cell)
        # plot_atom_positions1(h, h.ct.virtual_and_interfacial_atoms, radial_dep)

        for jj, item in enumerate(k_points):

            [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
            band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    ax = plt.axes()
    ax.plot(band_structure[ :, indices_of_bands ])
    ax.set_xlabel( "" )
    ax.set_ylabel( "Energy (eV)" )
    ax.set_title( "" )
    plt.tight_layout()
    plt.ylim((-1, 1))
    plt.show()

    k_index = np.linspace(0, 1, np.size(k_points, axis=0))
    band_structure_data = np.c_[k_index[:,None],band_structure]
    np.savetxt(path_to_data_file, np.c_[band_structure_data])

    # band_structure_data = np.c_[list_of_spin_orbit_couplings[:,None],band_structure]
    # np.savetxt(path_to_data_file, np.c_[band_structure_data])

if __name__ == '__main__':

    main()
