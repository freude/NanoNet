"""
The module contains functions computing hopping parameters
with arbitrary rotations of atomic orbitals based on the table of
empirical diatomic couplings defined in the module params.
Computations are based mostly on analytical equations derived in
A.V. Podolskiy and P. Vogl, Phys. Rev. B. 69, 233101 (2004)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import sys
import math
import numpy as np
from .p import *
import warnings


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
PARAMS_H_SI = {'ss_sigma': -3.9997,
               'cs_sigma': -1.6977,
               'sp_sigma': 4.2518,
               'sd_sigma': -2.1055}

PARAMS_H_H = {'ss_sigma': 1}


#  NN - Bi-Bi
PARAMS_BI_BI1 = {'ss_sigma': -0.608,
                 'sp_sigma': 1.320,
                 'pp_sigma': 1.854,
                 'pp_pi': -0.600}

# 2d NN - Bi-Bi
PARAMS_BI_BI2 = {'ss_sigma': -0.384,
                 'sp_sigma': 0.433,
                 'pp_sigma': 1.396,
                 'pp_pi': -0.344}

# 3d NN - Bi-Bi
PARAMS_BI_BI3 = {'ss_sigma': 0,
                 'sp_sigma': 0,
                 'pp_sigma': 0.156,
                 'pp_pi': 0}

def me_diatomic(bond, n, l_min, l_max, m, which_neighbour):
    """
    The function looks up into the table of parameters making a query parametrized by:

    :param bond:   a bond type represented by a list of atom labels
    :param n:      combination of the principal quantum numbers of atoms
    :param l_min:  min(l1, l2), where l1 and l2 are orbital quantum numbers of atoms
    :param l_max:  max(l1, l2), where l1 and l2 are orbital quantum numbers of atoms
    :param m:      symmetry of the electron wave function in the diatomic molecule
                   takes values "sigma", "pi" and "delta"
    :return:       numerical value of the corresponding tabular parameter
    :rtype:        float
    """

    if n == '00':
        label = ORBITAL_QN[l_min] + ORBITAL_QN[l_max] + '_' + M_QN[m]
    elif n == '01':
        label = 'c' + ORBITAL_QN[l_max] + '_' + M_QN[m]
    elif n == '11':
        label = 'cc' + '_' + M_QN[m]
    else:
        raise ValueError('Wrong value of the value variable')

    try:
        if which_neighbour == 0:
            return getattr(sys.modules[__name__], 'PARAMS_' + bond)[label]
        else:
            return getattr(sys.modules[__name__], 'PARAMS_' + bond + str(which_neighbour))[label]
    except KeyError:
        warnings.warn('There is no parameter PARAMS_' +
                      bond + '[' + label + ']' + ' in the dictionary', UserWarning)
        return 0


def d_me(N, l, m1, m2):
    """
    Computes rotational matrix elements according to
    A.V. Podolskiy and P. Vogl, Phys. Rev. B. 69, 233101 (2004)

    :param N:    directional cosine relative to z-axis
    :param l:    orbital quantum number
    :param m1:   magnetic quantum number
    :param m2:   magnetic quantum number
    :return:     rotational matrix element
    """

    prefactor = ((0.5 * (1 + N)) ** l) * ((old_div((1 - N), (1 + N))) ** (m1 * 0.5 - m2 * 0.5)) * \
        math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) *
                  math.factorial(l + m1) * math.factorial(l - m1))

    ans = 0

    for t in range(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            ans += ((-1) ** t) * ((old_div((1 - N), (1 + N))) ** t) / \
                   (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) *
                    math.factorial(t) * math.factorial(t + m1 - m2))

    return np.nan_to_num(ans * prefactor)


def tau(m):
    if m < 0:
        return 0
    else:
        return 1


def a_coef(m, gamma):

    if m == 0:
        return old_div(1.0, math.sqrt(2))
    else:
        return ((-1) ** abs(m)) * \
               (tau(m) * math.cos(abs(m) * gamma) - tau(-m) * math.sin(abs(m) * gamma))


def b_coef(m, gamma):

    return ((-1) ** abs(m)) * \
           (tau(m) * math.sin(abs(m) * gamma) + tau(-m) * math.cos(abs(m) * gamma))


def s_me(N, l, m1, m2, gamma):

    return a_coef(m1, gamma) * \
           (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) + d_me(N, l, abs(m1), -abs(m2)))


def t_me(N, l, m1, m2, gamma):

    if m1 == 0:
        return 0
    else:
        return b_coef(m1, gamma) * \
               (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) - d_me(N, l, abs(m1), -abs(m2)))


def me(atom1, ll1, atom2, ll2, coords, which_neighbour=0):

    # determine type of bonds
    atoms = sorted([item.upper() for item in [atom1.title, atom2.title]])
    atoms = atoms[0] + '_' + atoms[1]

    # quantum numbers for the first atom
    n1 = atom1.orbitals[ll1]['n']
    l1 = atom1.orbitals[ll1]['l']
    m1 = atom1.orbitals[ll1]['m']

    # quantum numbers for the second atom
    n2 = atom2.orbitals[ll2]['n']
    l2 = atom2.orbitals[ll2]['l']
    m2 = atom2.orbitals[ll2]['m']

    L = coords[0]
    M = coords[1]
    N = coords[2]

    gamma = math.atan2(L, M)

    code = str(min(n1, n2)) + str(max(n1, n2))

    l_min = min(l1, l2)
    l_max = max(l1, l2)

    prefactor = (-1) ** ((l1 - l2 + abs(l1 - l2)) * 0.5)
    ans = 2 * a_coef(m1, gamma) * a_coef(m2, gamma) * \
        d_me(N, l1, abs(m1), 0) * d_me(N, l2, abs(m2), 0) * \
        me_diatomic(atoms, code, l_min, l_max, 0, which_neighbour)

    for m in range(1, l_min+1):
        ans += (s_me(N, l1, m1, m, gamma) * s_me(N, l2, m2, m, gamma) +
                t_me(N, l1, m1, m, gamma) * t_me(N, l2, m2, m, gamma)) * \
               me_diatomic(atoms, code, l_min, l_max, m, which_neighbour)

    return prefactor * ans


if __name__ == "__main__":

    x0 = np.array([0, 0, 0], dtype=float)
    x1 = np.array([1, 1, 1], dtype=float)

    coords = x0 - x1
    coords /= np.linalg.norm(coords)

    print(coords)

    # print d_me(coords[2], 0, 0, 0)
    # print d_me(-coords[2], 0, 0, 0)
    # print d_me(-coords[2], 1, 0, 0)

    print(d_me(-coords[2], 1, 1, 0))
    print(d_me(-coords[2], 1, 0, 1))
    print(d_me(-coords[2], 2, 1, 0))
    print(d_me(-coords[2], 2, 0, 1))
    print(d_me(-coords[2], 2, 2, 1))
    print(d_me(-coords[2], 2, 1, 2))
    print("-----------------------------")
    print(d_me(coords[2], 1, 1, 0))
    print(d_me(coords[2], 1, 0, 1))
    print(d_me(coords[2], 2, 1, 0))
    print(d_me(coords[2], 2, 0, 1))
    print(d_me(coords[2], 2, 2, 1))
    print(d_me(coords[2], 2, 1, 2))
    # print d_me(-coords[2], 1, -1, 0)
    # print d_me(-coords[2], 1, 0, -1)
    # print d_me(-coords[2], 1, -1, -1)
    # print d_me(-coords[2], 1, 1, 1)
