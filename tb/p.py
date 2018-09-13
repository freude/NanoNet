"""
This module contains the input parameters for performing calculations of band structure using empirical tight binding theory.
"""
from __future__ import division
from past.utils import old_div
import numpy as np


# Mathematical constants
PI = 3.141592653589793238462643383279502884197169399375105820974


# Lattice constants
a_si = 5.50
a_bi = 4.5332
c_bi = 11.7967
b_bi = a_bi / c_bi
gamma_bi = 0.2303


# Labels for orbitals and bonds
ORBITAL_QN = {0: 's', 1: 'p', 2: 'd'}
M_QN = {0: 'sigma', 1: 'pi', 2: 'delta'}


# Coordinates of points with high symmetry in the first Brillouin zone

SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [old_div(PI, a_si),  old_div(PI, a_si),  old_div(PI, a_si)],
    'W': [old_div(PI, a_si),  2 * PI / a_si,  0],
    'U': [old_div(PI, (2 * a_si)), 2 * PI / a_si, old_div(PI, (2 * a_si))],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}

recip_lat_basis = np.matrix([[-1, -np.sqrt(3.0)/3.0, b_bi],
                             [1, -np.sqrt(3.0)/3.0, b_bi],
                             [0, 2*np.sqrt(3.0)/3.0, b_bi]]) * 2 * PI / a_bi

SPECIAL_K_POINTS_BI = {
    'GAMMA': (recip_lat_basis * np.matrix([0, 0, 0]).T).tolist(),
    'X': (recip_lat_basis * np.matrix([0.5, 0.5, 0.0]).T).tolist(),
    'T': (recip_lat_basis * np.matrix([0.5, 0.5, 0.5]).T).tolist(),
    'L': (recip_lat_basis * np.matrix([0.0, 0.5, 0.0]).T).tolist(),
    'W': (recip_lat_basis * np.matrix([gamma_bi, 1.0-gamma_bi, 0.5]).T).tolist(),
    'K': (recip_lat_basis * np.matrix([old_div(1.0,4)+0.5*gamma_bi, old_div(3.0,4)-0.5*gamma_bi, 0.0]).T).tolist(),
    'U': (recip_lat_basis * np.matrix([0.5*gamma_bi+0.25, 1.0-gamma_bi, 0.5*gamma_bi+0.25]).T).tolist()
}
