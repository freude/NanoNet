"""
The module contains functions computing hopping parameters
with arbitrary rotations of atomic orbitals based on the table of
empirical diatomic couplings defined in the module params.
Computations are based mostly on analytical equations derived in
[A.V. Podolskiy and P. Vogl, Phys. Rev. B. 69, 233101 (2004)].
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sys
import math
from .constants import *
from nanonet.tb import tb_params


def me_diatomic(bond, n, l_min, l_max, m, which_neighbour, overlap=False):
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

    label = n[0] + ORBITAL_QN[l_min] + n[1] + ORBITAL_QN[l_max] + '_' + M_QN[m]

    # if n == '00':
    #     label = ORBITAL_QN[l_min] + ORBITAL_QN[l_max] + '_' + M_QN[m]
    # elif n == '01':
    #     label = 'c' + ORBITAL_QN[l_max] + '_' + M_QN[m]
    # elif n == '11':
    #     label = 'cc' + '_' + M_QN[m]
    # else:
    #     raise ValueError('Wrong value of the value variable')

    if overlap:
        flag = 'OV_'
    else:
        flag = 'PARAMS_'

    try:
        if which_neighbour == 0:
            return getattr(sys.modules[tb_params.__name__], flag + bond)[label]
        elif which_neighbour == 100:
            return 0
        else:
            return getattr(sys.modules[tb_params.__name__], flag + bond + str(which_neighbour))[label]
    except KeyError:
        return 0


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
        prefactor = math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) *
                              math.factorial(l + m1) * math.factorial(l - m1))
    else:
        prefactor = ((0.5 * (1 + N)) ** l) * (((1 - N) / (1 + N)) ** (m1 * 0.5 - m2 * 0.5)) * \
            math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) *
                      math.factorial(l + m1) * math.factorial(l - m1))

    ans = 0
    for t in range(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            if N == -1.0 and t == 0:
                ans += ((-1) ** t) / \
                       (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) *
                        math.factorial(t) * math.factorial(t + m1 - m2))
            else:
                ans += ((-1) ** t) * (((1 - N) / (1 + N)) ** t) / \
                       (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) *
                        math.factorial(t) * math.factorial(t + m1 - m2))

    return np.nan_to_num(ans * prefactor)


def tau(m):
    """

    Parameters
    ----------
    m :
        

    Returns
    -------

    """
    if m < 0:
        return 0
    else:
        return 1


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
        return 1.0 / math.sqrt(2)
    else:
        return ((-1) ** abs(m)) * \
               (tau(m) * math.cos(abs(m) * gamma) - tau(-m) * math.sin(abs(m) * gamma))


def b_coef(m, gamma):
    """

    Parameters
    ----------
    m :
        
    gamma :
        

    Returns
    -------

    """

    return ((-1) ** abs(m)) * \
           (tau(m) * math.sin(abs(m) * gamma) + tau(-m) * math.cos(abs(m) * gamma))


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

        L = coords[0]
        M = coords[1]
        N = coords[2]

        gamma = math.atan2(L, M)

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
            me_diatomic(atoms, code, l_min, l_max, 0, which_neighbour, overlap=overlap)

        for m in range(1, l_min+1):
            ans += (s_me(N, l1, m1, m, gamma) * s_me(N, l2, m2, m, gamma) +
                    t_me(N, l1, m1, m, gamma) * t_me(N, l2, m2, m, gamma)) * \
                   me_diatomic(atoms, code, l_min, l_max, m, which_neighbour, overlap=overlap)

        return prefactor * ans
    else:
        return 0


def me_qsymm(atom1, ll1, atom2, ll2, coords):
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
    family = tb_params.family
    params = tb_params.params

    # determine type of bonds
    atoms = sorted([item.upper() for item in [atom1.title, atom2.title]])
    atoms = atoms[0] + '_' + atoms[1]

    # quantum numbers for the first atom
    n1 = atom1.orbitals[ll1]['tag']
    s1 = atom1.orbitals[ll1]['s']

    # quantum numbers for the second atom
    n2 = atom2.orbitals[ll2]['tag']
    s2 = atom2.orbitals[ll2]['s']

    if s1 == s2:
        norm = np.linalg.norm(coords)
        L = coords[0] / norm
        M = coords[1] / norm
        N = coords[2] / norm

        ans = list(family[0].data.values())[0] * 0.0

        for j, item in enumerate(family):
            keys = list(item.data.keys())
            values = list(item.data.values())
            for j1, item1 in enumerate(keys):
                item1 = np.hstack((item1[0], np.zeros(3)))
                if np.abs(L - item1[0]) < 0.001 and\
                   np.abs(M - item1[1]) < 0.001 and\
                   np.abs(N - item1[2]) < 0.001:
                    ans += params[j] * values[j1]
                    # print(L, item1[0][0])
        print(ans[n1, n2])
        return ans[n1, n2]
    else:
        return 0


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
