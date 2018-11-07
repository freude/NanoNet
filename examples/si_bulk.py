import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
import examples.data_bi_bulk
from examples import data_bi_bulk
from tb.plotting import plot_bs_split, plot_atom_positions


def radial_dep(coords):

    norm_of_coords = np.linalg.norm(coords)
    print(norm_of_coords)
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 3
    else:
        return 100


def main1():

    from tb import get_k_coords

    path_to_xyz_file = 'input_samples/bulk_silicon.xyz'
    species = 'Si'
    basis_set = 'SiliconSP3D5S'
    sym_points = [ 'L', 'GAMMA', 'X' ]

    num_points = [ 20, 20 ]
    indices_of_bands = range( 0, 8 )

    primitive_cell = examples.data_bi_bulk.a_si * np.array([[0.0, 0.5, 0.5],
                                                            [0.5, 0.0, 0.5],
                                                            [0.5, 0.5, 0.0]])

    Atom.orbital_sets = { species: basis_set }

    h = Hamiltonian( xyz = path_to_xyz_file)
    h.initialize()
    h.set_periodic_bc( primitive_cell )

    k_points = get_k_coords( sym_points, num_points, species )

    band_structure = []
    for jj, item in enumerate( k_points ):
        [ eigenvalues, _ ] = h.diagonalize_periodic_bc( k_points[ jj ] )
        band_structure.append( eigenvalues )

    band_structure = np.array( band_structure )

    ax = plt.axes()
    ax.plot( band_structure[ :, indices_of_bands ] )
    ax.set_xlabel( "" )
    ax.set_ylabel( "Energy (eV)" )
    ax.set_title( "" )
    plt.tight_layout()
    plt.show()
    # plt.savefig( path_to_pdf_file )


def main2():

    from tb.aux_functions import get_k_coords

    path_to_xyz_file = 'input_samples/bulk_bismuth.xyz'
    # path_to_pdf_file = '../band_structure_of_bulk_bismuth.pdf'
    species = 'Bi'
    basis_set = 'Bismuth'
    sym_points = ['K', 'GAMMA', 'T', 'W', 'L', 'LAMBDA']
    # sym_points = ['K', 'X', 'GAMMA', 'L', 'U', 'T']
    # sym_points = ['GAMMA', 'GAMMA']
    # sym_points = ['K', 'GAMMA', 'T']

    num_points = [20, 20, 20, 20, 20]
    # num_points = [1]
    # num_points = [20, 20]
    indices_of_bands = range(0, 16)

    cell_a = examples.data_bi_bulk.a_bi * np.array([[(-1.0 / 2.0), (-np.sqrt(3.0) / 6.0), 0.0],
                                                    [ (  1.0 / 2.0 ), ( -np.sqrt(3.0) / 6.0 ), 0.0 ],
                                                    [ 0.0,            (  np.sqrt(3.0) / 3.0 ), 0.0 ]])
    cell_c = examples.data_bi_bulk.c_bi * np.array([[0.0, 0.0, (1.0 / 3.0)],
                                                    [ 0.0, 0.0, ( 1.0 / 3.0 ) ],
                                                    [ 0.0, 0.0, ( 1.0 / 3.0 ) ]])
    primitive_cell = cell_a + cell_c

    Atom.orbital_sets = { species: basis_set }

    h = Hamiltonian( xyz = path_to_xyz_file, nn_distance = 4.6, so_coupling=1.2)
    h.initialize( radial_dep )
    h.set_periodic_bc(primitive_cell.tolist())

    k_points = get_k_coords( sym_points, num_points, species )

    band_structure = []
    for jj, item in enumerate( k_points ):
        [ eigenvalues, _ ] = h.diagonalize_periodic_bc( k_points[ jj ] )
        band_structure.append( eigenvalues )

    band_structure = np.array( band_structure )

    print(h.is_hermitian())

    ax = plt.axes()
    ax.plot( band_structure[ :, indices_of_bands ] )
    ax.set_xlabel( "" )
    ax.set_ylabel( "Energy (eV)" )
    ax.set_title( "" )
    plt.tight_layout()
    plt.show()
    # plt.savefig( path_to_pdf_file )

    # plt.imshow(np.abs(h.h_matrix_bc_factor * h.h_matrix + h.h_matrix_bc_add))
    # plt.show()

def main3():
    """
    This function computes the band gap / band structure for Silicon Nanowire
    :param path: directory path to where xyz input files are stored
    :param flag: boolean statements
    :return: band gap / band structure
    """
    # define orbitals sets
    Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    band_gaps = []
    band_structures = []

    path = "./input_samples/SiNW2.xyz"

    hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4)
    hamiltonian.initialize()

    if True:
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

    for jj in range(num_points):
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

    if True:
        plot_bs_split(kk, vba, cba)


def main4():

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

    primitive_cell = data_bi_bulk.cell

    Atom.orbital_sets = { species: basis_set }

    h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=4.7, so_coupling=0.9)
    h.initialize(radial_dep)
    h.set_periodic_bc(primitive_cell)
    plot_atom_positions(h.atom_list, h.ct.virtual_and_interfacial_atoms, radial_dep)

    k_points = get_k_coords( sym_points, num_points, species )

    list_of_spin_orbit_couplings = [0.27]
    # list_of_spin_orbit_couplings = np.linspace(0, 0.333333, 40)

    band_structure = []
    for ii, item in enumerate(list_of_spin_orbit_couplings):
        for jj, item in enumerate(k_points):
            # h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=5.6)
            # h.initialize(radial_dep)
            # h.set_periodic_bc(primitive_cell, radial_dep)data_bi_bulk.py
            data_bi_bulk.LAMBDA = list_of_spin_orbit_couplings[ii]
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

    main2()
