import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
from examples import data_bi_bilayer
from tb.plotting import plot_atom_positions
import tb


bi = Atom('Bi')
bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=0)
bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=0)
bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=0)
bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=0)
bi.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=1)
bi.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=1)
bi.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=1)
bi.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=1)


def radial_dep(coords):

    norm_of_coords = np.linalg.norm(coords)

    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 100
    else:
        return 100


def main():

    from tb.aux_functions import get_k_coords

    path_to_xyz_file = """2
                              Bilayer Bismuth
                              Bi   0.000000    2.499364    0.868720
                              Bi   2.164513    1.249682   -0.868720
                           """

    # path_to_pdf_file = '../band_structure_of_bulk_bismuth.pdf'
    species = 'Bi'
    basis_set = 'Bismuth'
    sym_points = ['M', 'GAMMA', 'K']
    # sym_points = ['GAMMA', 'GAMMA']

    num_points = [40, 40]
    # num_points = [1]
    indices_of_bands = range( 0, 16 )

    primitive_cell = data_bi_bilayer.cell

    Atom.orbital_sets = {species: bi}

    tb.set_tb_params(PARAMS_BI_BI1=data_bi_bilayer.PARAMS_BI_BI1,
                     PARAMS_BI_BI2=data_bi_bilayer.PARAMS_BI_BI2,
                     PARAMS_BI_BI3=data_bi_bilayer.PARAMS_BI_BI3)

    h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=1.8)
    h.initialize(radial_dep)
    h.set_periodic_bc(primitive_cell)
    plot_atom_positions(h.atom_list, h.ct.virtual_and_interfacial_atoms, radial_dep)

    k_points = get_k_coords( sym_points, num_points, data_bi_bilayer.SPECIAL_K_POINTS_BI)

    # list_of_spin_orbit_couplings = [0.27]
    # list_of_spin_orbit_couplings = np.linspace(0, 0.333333, 40)

    band_structure = []
    # for ii, item in enumerate(list_of_spin_orbit_couplings):
    for jj, item in enumerate(k_points):
        # h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=5.6)
        # h.initialize(radial_dep)
        # h.set_periodic_bc(primitive_cell, radial_dep)data_bi_bulk.py
        # data_bi_bilayer.LAMBDA = list_of_spin_orbit_couplings[ii]
        [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
        band_structure.append(eigenvalues)

    band_structure = np.array(band_structure)

    ax = plt.axes()
    ax.plot(band_structure[ :, indices_of_bands ])
    ax.set_xlabel( "" )
    ax.set_ylabel( "Energy (eV)" )
    ax.set_title( "" )
    plt.tight_layout()
    plt.ylim((-2, 2))
    plt.show()


if __name__ == '__main__':

    main()
