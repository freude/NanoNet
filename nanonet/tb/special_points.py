import numpy as np


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    """

    Parameters
    ----------
    reciprocal_lattice_vectors :
        
    coordinate_of_high_symmetry_point :
        

    Returns
    -------

    """
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.dot(coordinate_of_high_symmetry_point,
                                                                 reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


PI = np.pi

# Lattice constants
a_si = 5.50
a_bi = 4.5332
c_bi = 11.7967
g_bi = 1.3861
gamma_bi = 0.240385652727133

SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [PI / a_si,  PI / a_si,  PI/ a_si],
    'W': [PI / a_si,  2 * PI / a_si,  0],
    'U': [PI / (2 * a_si), 2 * PI / a_si, PI/ (2 * a_si)],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}


reciprocal_lattice_vectors_bi = g_bi * np.array([[-1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                  [1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                  [0.0, (2.0 * np.sqrt(3.0) / 3.0), (a_bi / c_bi)]])

SPECIAL_K_POINTS_BI = {
    'LAMBDA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.25, 0.25, 0.25]),
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.00, 0.00]),
    'T': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.50]),
    'L': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.50, 0.00]),
    'X': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.00]),
    'K': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (0.75 -
                                                                                    (0.50 * gamma_bi)), 0.00]),
    'W': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, (1.00 - gamma_bi), gamma_bi]),
    'U': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (1.00 -
                                                                                    gamma_bi), ((0.50 * gamma_bi) +
                                                                                    0.25)])
}