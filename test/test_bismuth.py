import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
import p


def radial_dep( coords ):
    norm_of_coords = np.linalg.norm( coords )
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif norm_of_coords > 3.7:
        return 3


def main1():

    from tb import get_k_coords

    path_to_xyz_file = 'input_samples/bulk_silicon.xyz'
    species = 'Si'
    basis_set = 'SiliconSP3D5S'
    sym_points = [ 'L', 'GAMMA', 'X' ]

    num_points = [ 20, 20 ]
    indices_of_bands = range( 0, 8 )

    primitive_cell = p.a_si * np.array( [ [0.0, 0.5, 0.5],
                                          [0.5, 0.0, 0.5],
                                          [0.5, 0.5, 0.0] ] )

    Atom.orbital_sets = { species: basis_set }

    h = Hamiltonian( xyz = path_to_xyz_file )
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

    num_points = [20, 20, 20, 20, 20]
    # num_points = [1]
    indices_of_bands = range( 0, 8 )

    cell_a = p.a_bi * np.array( [ [ ( -1.0 / 2.0 ), ( -np.sqrt(3.0) / 6.0 ), 0.0 ],
                                  [ (  1.0 / 2.0 ), ( -np.sqrt(3.0) / 6.0 ), 0.0 ],
                                  [ 0.0,            (  np.sqrt(3.0) / 3.0 ), 0.0 ] ] )
    cell_c = p.c_bi * np.array( [ [ 0.0, 0.0, ( 1.0 / 3.0 ) ],
                                  [ 0.0, 0.0, ( 1.0 / 3.0 ) ],
                                  [ 0.0, 0.0, ( 1.0 / 3.0 ) ] ] )
    primitive_cell = cell_a + cell_c

    Atom.orbital_sets = { species: basis_set }

    h = Hamiltonian( xyz = path_to_xyz_file, nn_distance = 5.0 )
    h.initialize( radial_dep )
    h.set_periodic_bc( primitive_cell.tolist() )

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


if __name__ == '__main__':

    main2()
