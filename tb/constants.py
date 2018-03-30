"""
The module contains empirical tight-binding parameters and auxiliary global variables.
"""

PI = 3.14159

# The codes for labeling the orbitals

ORBITAL_QN = {0: 's', 1: 'p', 2: 'd'}

M_QN = {0: 'sigma', 1: 'pi', 2: 'delta'}

# Coordinates of high-symmetry points in the Brillouin zone
# of the diamond-type crystal lattice

a_si = 5.50

SPECIAL_K_POINTS = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [PI / a_si,  PI / a_si,  PI / a_si],
    'W': [PI / a_si,  2 * PI / a_si,  0],
    'U': [PI / (2 * a_si), 2 * PI / a_si, PI / (2 * a_si)],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}
