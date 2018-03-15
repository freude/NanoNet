"""
The module contains functions computing hopping parameters
with arbitrary rotations of atomic orbitals based on the table of
empirical diatomic couplings defined in the module params.
Computations are based mostly on analytical equations derived in
A.V. Podolskiy and P. Vogl, Phys. Rev. B. 69, 233101 (2004)
"""

__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"
__version__ = "0.0.1"

import math
import numpy as np
from params import *


def me_diatomic(atoms, n, l_min, l_max, m):
    """
    The function looks up into the table of parameters taking a query parametrized by:

    :param atoms:  type of bound , it may be "Si-Si" or "Si-H" bound
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

    if atoms == "Si-Si":
        return PARAMS_SI_SI[label]
    elif atoms == "Si-H":
        return PARAMS_SI_H[label]
    else:
        raise ValueError('Wrong value of the iter variable')


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

    prefactor = ((0.5 * (1 + N)) ** l) * (((1 - N) / (1 + N)) ** (m1 * 0.5 - m2 * 0.5)) * \
        math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) *
                  math.factorial(l + m1) * math.factorial(l - m1))

    ans = 0

    for t in xrange(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            ans += ((-1) ** t) * (((1 - N) / (1 + N)) ** t) / \
                   (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) *
                    math.factorial(t) * math.factorial(t + m1 - m2))

    return ans * prefactor


def tau(m):
    if m < 0:
        return 0
    else:
        return 1


def a_coef(m, gamma):

    if m == 0:
        return 1.0 / math.sqrt(2)
    else:
        return ((-1) ** abs(m)) * (tau(m) * math.cos(abs(m) * gamma) - tau(-m) * math.sin(abs(m) * gamma))


def b_coef(m, gamma):

    return ((-1) ** abs(m)) * (tau(m) * math.sin(abs(m) * gamma) + tau(-m) * math.cos(abs(m) * gamma))


def s_me(N, l, m1, m2, gamma):

    return a_coef(m1, gamma) * (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) + d_me(N, l, abs(m1), -abs(m2)))


def t_me(N, l, m1, m2, gamma):

    if m1 == 0:
        return 0
    else:
        return b_coef(m1, gamma) * (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) - d_me(N, l, abs(m1), -abs(m2)))


def me(atoms, n1, l1, m1, n2, l2, m2, coords):

    L = coords[0]
    M = coords[1]
    N = coords[2]

    print atoms, n1, l1, m1, n2, l2, m2, N, M

    # gamma = math.asin(M / math.sqrt(1.0 - N ** 2))
    gamma = math.atan2(L, M)

    code = str(min(n1, n2)) + str(max(n1, n2))

    l_min = min(l1, l2)
    l_max = max(l1, l2)

    prefactor = (-1) ** ((l1 - l2 + abs(l1 - l2)) * 0.5)
    ans = 2 * a_coef(m1, gamma) * a_coef(m2, gamma) * \
          d_me(N, l1, abs(m1), 0) * d_me(N, l2, abs(m2), 0) * me_diatomic(atoms, code, l_min, l_max, 0)

    for m in xrange(1, l_min+1):
        ans += (s_me(N, l1, m1, m, gamma) * s_me(N, l2, m2, m, gamma) +
                t_me(N, l1, m1, m, gamma) * t_me(N, l2, m2, m, gamma)) * \
               me_diatomic(atoms, code, l_min, l_max, m)

    return prefactor * ans


if __name__ == "__main__":

    x0 = np.array([0, 0, 0], dtype=float)
    x1 = np.array([1, 1, 1], dtype=float)

    coords = x0 - x1
    coords /= np.linalg.norm(coords)

    print coords

    # print d_me(coords[2], 0, 0, 0)
    # print d_me(-coords[2], 0, 0, 0)
    # print d_me(-coords[2], 1, 0, 0)

    print d_me(-coords[2], 1, 1, 0)
    print d_me(-coords[2], 1, 0, 1)
    print d_me(-coords[2], 2, 1, 0)
    print d_me(-coords[2], 2, 0, 1)
    print d_me(-coords[2], 2, 2, 1)
    print d_me(-coords[2], 2, 1, 2)
    print "-----------------------------"
    print d_me(coords[2], 1, 1, 0)
    print d_me(coords[2], 1, 0, 1)
    print d_me(coords[2], 2, 1, 0)
    print d_me(coords[2], 2, 0, 1)
    print d_me(coords[2], 2, 2, 1)
    print d_me(coords[2], 2, 1, 2)
    # print d_me(-coords[2], 1, -1, 0)
    # print d_me(-coords[2], 1, 0, -1)
    # print d_me(-coords[2], 1, -1, -1)
    # print d_me(-coords[2], 1, 1, 1)