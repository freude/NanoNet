"""
The module contains a symbolic version of the diatomic matrix elements computing.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import math
from sympy.core import *
from .constants import *
from sympy import cos, sqrt, sin, atan2, Abs, factorial
from nanonet.tb.diatomic_matrix_element import me_diatomic, tau


def to_symbol(value):
    strline = "{:.3f}".format(np.abs(value))
    if value < 0:
        value = -Symbol(strline)
    elif value == 0:
        value = 0
    elif value == 1.0:
        value = 1.0
    else:
        value = Symbol(strline)

    return value


def me_diatomic_sym(bond, n, l_min, l_max, m, which_neighbour, overlap=False):
    """The function looks up into the table of parameters making a query parametrized by:

    Parameters
    ----------
    bond :
        a bond type represented by a list of atom labels
    n :
        combination of the principal quantum numbers of atoms
    l_min :
        min(l1, l2), where l1 and l2 are orbital quantum numbers of atoms
    l_max :
        max(l1, l2), where l1 and l2 are orbital quantum numbers of atoms
    m :
        symmetry of the electron wave function in the diatomic molecule
        takes values "sigma", "pi" and "delta"
    which_neighbour :

    overlap : bool
        A flag indicating that the overlap matrix element has to be computed
        

    Returns
    -------
    float
        numerical value of the corresponding tabular parameter

    """

    val = me_diatomic(bond, n, l_min, l_max, m, which_neighbour, overlap=overlap)

    if val != 0:

        label = n[0] + ORBITAL_QN[l_min] + n[1] + ORBITAL_QN[l_max] + '_' + M_QN[m]

        if overlap:
            flag = 'ov'
        else:
            flag = ''

        if which_neighbour == 0:
            return Symbol(flag + ''.join(bond.split('_')) + '_' + label, real=True)
        else:
            return Symbol(flag + ''.join(bond.split('_')) + str(which_neighbour) + '_' + label, real=True)

    else:
        return val


def d_me(N, l, m1, m2):
    """Computes rotational matrix elements according to
    A.V. Podolskiy and P. Vogl, Phys. Rev. B. 69, 233101 (2004)

    Parameters
    ----------
    N :
        directional cosine relative to z-axis
    l :
        orbital quantum number
    m1 :
        magnetic quantum number
    m2 :
        magnetic quantum number

    Returns
    -------
    type
        rotational matrix element

    """

    if N == -1.0 and m1 == m2:
        prefactor = sqrt(factorial(l + m2) * factorial(l - m2) * factorial(l + m1) * factorial(l - m1))
    else:
        prefactor = ((0.5 * (1 + N)) ** l) * (((1 - N) / (1 + N)) ** (m1 * 0.5 - m2 * 0.5)) * \
            sqrt(factorial(l + m2) * factorial(l - m2) * factorial(l + m1) * factorial(l - m1))

    ans = 0
    for t in range(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            if N == -1.0 and t == 0:
                ans += ((-1) ** t) / \
                       (factorial(l + m2 - t) * factorial(l - m1 - t) * factorial(t) * factorial(t + m1 - m2))
            else:
                ans += ((-1) ** t) * (((1 - N) / (1 + N)) ** t) / \
                       (factorial(l + m2 - t) * factorial(l - m1 - t) * factorial(t) * factorial(t + m1 - m2))

    return ans * prefactor


def a_coef(m, gamma):
    """

    Parameters
    ----------
    m :

    gamma :


    Returns
    -------

    """

    if m == 0:
        return 1 / sqrt(2)
    else:
        return ((-1) ** Abs(m)) * \
               (tau(m) * cos(Abs(m) * gamma) - tau(-m) * sin(Abs(m) * gamma))


def b_coef(m, gamma):
    """

    Parameters
    ----------
    m :

    gamma :


    Returns
    -------

    """

    return ((-1) ** Abs(m)) * \
           (tau(m) * sin(Abs(m) * gamma) + tau(-m) * cos(Abs(m) * gamma))


def s_me(N, l, m1, m2, gamma):
    """

    Parameters
    ----------
    N :

    l :

    m1 :

    m2 :

    gamma :


    Returns
    -------

    """

    return a_coef(m1, gamma) * \
           (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) + d_me(N, l, abs(m1), -abs(m2)))


def t_me(N, l, m1, m2, gamma):
    """

    Parameters
    ----------
    N :

    l :

    m1 :

    m2 :

    gamma :


    Returns
    -------

    """

    if m1 == 0:
        return 0
    else:
        return b_coef(m1, gamma) * \
               (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) - d_me(N, l, abs(m1), -abs(m2)))


def me(atom1, ll1, atom2, ll2, coords, which_neighbour=0, overlap=False):
    """Computes the non-diagonal matrix element of the tight-binding Hamiltonian -
    coupling between two sites, both are described by LCAO basis sets.
    This function is evoked in the member function _get_me() of the Hamiltonian object.

    Parameters
    ----------
    atom1 : tb.Orbitals
        basis set associated with the first site
    ll1 : int
        index specifying a particular orbital in the basis set for the first site
    atom2 : tb.Orbitals
        basis set associated with the first site
    ll2 : int
        index specifying a particular orbital in the basis set for the second site
    coords : array
        coordinates of radius vector pointing from one site to another
    which_neighbour : int
        Order of a nearest neighbour (first-, second-, third- etc) (Default value = 0)
    overlap : bool
            A flag indicating that the overlap matrix element has to be computed
    Returns
    -------

    
    """

    # determine type of bonds
    atoms = sorted([item.upper() for item in [atom1.title, atom2.title]])
    atoms = atoms[0] + '_' + atoms[1]

    # quantum numbers for the first atom
    n1 = atom1.orbitals[ll1]['n']
    l1 = atom1.orbitals[ll1]['l']
    m1 = atom1.orbitals[ll1]['m']
    s1 = atom1.orbitals[ll1]['s']

    # quantum numbers for the second atom
    n2 = atom2.orbitals[ll2]['n']
    l2 = atom2.orbitals[ll2]['l']
    m2 = atom2.orbitals[ll2]['m']
    s2 = atom2.orbitals[ll2]['s']

    if s1 == s2:

        L = to_symbol(coords[0])
        M = to_symbol(coords[1])
        N = to_symbol(coords[2])

        gamma = atan2(L, M)

        if l1 > l2:
            code = [n2, n1]
        elif l1 == l2:
            code = [min(n1, n2), max(n1, n2)]
        else:
            code = [n1, n2]

        for j, item in enumerate(code):
            if item == 0:
                code[j] = ""
            else:
                code[j] = str(item)

        l_min = min(l1, l2)
        l_max = max(l1, l2)

        prefactor = (-1) ** ((l1 - l2 + abs(l1 - l2)) * 0.5)
        ans = 2 * a_coef(m1, gamma) * a_coef(m2, gamma) * \
            d_me(N, l1, abs(m1), 0) * d_me(N, l2, abs(m2), 0) * \
            me_diatomic_sym(atoms, code, l_min, l_max, 0, which_neighbour, overlap=overlap)

        for m in range(1, l_min+1):
            ans += (s_me(N, l1, m1, m, gamma) * s_me(N, l2, m2, m, gamma) +
                    t_me(N, l1, m1, m, gamma) * t_me(N, l2, m2, m, gamma)) * \
                   me_diatomic_sym(atoms, code, l_min, l_max, m, which_neighbour, overlap=overlap)

        return prefactor * ans
    else:
        return 0

