import numpy as np
import matplotlib.pyplot as plt
from tb import Hamiltonian
from tb import Atom
import examples.data_si_bulk


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


def main():

    from tb import get_k_coords

    path_to_xyz_file = 'input_samples/bulk_silicon.xyz'
    species = 'Si'
    basis_set = 'SiliconSP3D5S'
    sym_points = [ 'L', 'GAMMA', 'X' ]

    num_points = [20, 20]
    indices_of_bands = range( 0, 8 )

    primitive_cell = examples.data_si_bulk.a * np.array([[0.0, 0.5, 0.5],
                                                         [0.5, 0.0, 0.5],
                                                         [0.5, 0.5, 0.0]])

    Atom.orbital_sets = {species: basis_set}

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


if __name__ == '__main__':

    main()
