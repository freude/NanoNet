"""
The module contains empirical tight-binding parameters and auxiliary global variables.
"""

__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"
__version__ = "0.0.1"


PI = 3.14159

VERBOSITY = 3

a_si = 5.50
PRIMITIVE_CELL = [[0,          0.5 * a_si, 0.5 * a_si],
                  [0.5 * a_si,      0,     0.5 * a_si],
                  [0.5 * a_si, 0.5 * a_si,     0]]

# Tight-binding empirical parameters taken from
# Jancu et al., Phys. Rev. B, 57, 6493 (1998) and
# Zheng et al., IEEE Trans. Electron Dev., 52, 1092 (2005))

#  On-site Si
PARAMS_ON_SITE_SI = {'s': -2.0196,
                     'c': 19.6748,
                     'p': 4.5448,
                     'd': 14.1836}

#  On-site H
PARAMS_ON_SITE_H = {'s': 0.9998}

#  NN - Si-Si
PARAMS_SI_SI = {'ss_sigma': -1.9413,
                'cc_sigma': -3.3081,
                'cs_sigma': -1.6933,
                'sp_sigma': 2.7836,
                'cp_sigma': 2.8428,
                'sd_sigma': -2.7998,
                'cd_sigma': -0.7003,
                'pp_sigma': 4.1068,
                'pp_pi': -1.5934,
                'pd_sigma': -2.1073,
                'pd_pi': 1.9977,
                'dd_sigma': -1.2327,
                'dd_pi': 2.5145,
                'dd_delta': -2.4734}

#  NN - Si-H
PARAMS_SI_H = {'ss_sigma': -3.9997,
               'cs_sigma': -1.6977,
               'sp_sigma': 4.2518,
               'sd_sigma': -2.1055}


# The codes for labeling the orbitals

ORBITALS_CODE = {0: 's', 1: 'c',
                 2: 'px', 3: 'py', 4: 'pz',
                 5: 'dz2', 6: 'dxz', 7: 'dyz', 8: 'dxy', 9: 'dx2my2'}

ORBITALS_CODE_N = {0: 0, 1: 1,
                   2: 0, 3: 0, 4: 0,
                   5: 0, 6: 0, 7: 0, 8: 0, 9: 0}


ORBITALS_CODE_L = {0: 0, 1: 0,
                   2: 1, 3: 1, 4: 1,
                   5: 2, 6: 2, 7: 2, 8: 2, 9: 2}


ORBITALS_CODE_M = {0: 0, 1: 0,
                   2: -1, 3: 1, 4: 0,
                   5: -1, 6: -2, 7: 2, 8: 1, 9: 0}

ORBITAL_QN = {0: 's', 1: 'p', 2: 'd'}

M_QN = {0: 'sigma', 1: 'pi', 2: 'delta'}


# Coordinates of high-symmetry points in the Brillouin zone
# of the diamond-type crystal lattice

SPECIAL_K_POINTS = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [PI / a_si,  PI / a_si,  PI / a_si],
    'W': [PI / a_si,  2 * PI / a_si,  0],
    'U': [PI / (2 * a_si), 2 * PI / a_si, PI / (2 * a_si)],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}
