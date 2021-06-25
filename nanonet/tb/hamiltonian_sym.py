"""
The module contains a library of classes facilitating computations of Hamiltonian matrices.
"""
from __future__ import print_function, division
from __future__ import absolute_import
import numpy as np
from sympy.core import *
from sympy import latex
from sympy import exp, simplify, pprint, init_printing, nsimplify
from sympy.matrices import zeros, ones, matrix_multiply_elementwise
from nanonet.tb.diatomic_matrix_element_sym import me
from nanonet.tb.orbitals import Orbitals
import nanonet.verbosity as verbosity
from nanonet.tb.hamiltonian import Hamiltonian
from nanonet.tb.hamiltonian import unique_distances
from sympy import init_printing


init_printing(use_unicode=True)


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


class HamiltonianSym(Hamiltonian):
    """Class defines a Hamiltonian matrix in the symbolic representation.
    """

    def __init__(self, **kwargs):
        super(HamiltonianSym, self).__init__(**kwargs)
        if self.so_coupling != 0.0:
            self.so_coupling = Symbol('SO')

    def _initialize(self):
        self._coords = [0 for _ in range(self.basis_size)]
        # initialize Hamiltonian matrices
        self.h_matrix = zeros(self.basis_size, self.basis_size)
        self.h_matrix_bc_add = zeros(self.basis_size, self.basis_size)
        self.h_matrix_bc_factor = ones(self.basis_size, self.basis_size)

        if self.compute_overlap:
            self.ov_matrix = zeros(self.basis_size, self.basis_size)
            self.ov_matrix_bc_add = zeros(self.basis_size, self.basis_size)

    def initialize(self, int_radial_dep=None, radial_dep=None):
        """Compute matrix elements of the Hamiltonian.

        Parameters
        ----------
        int_radial_dep : func
             Integer radial dependence function (Default value = None)
        radial_dep : func
             Radial dependence function (Default value = None)

        Returns
        -------
        type Hamiltonian
            Returns the instance of the class Hamiltonian
        """

        super().initialize(int_radial_dep=int_radial_dep, radial_dep=radial_dep)

    def _get_me(self, atom1, atom2, l1, l2, coords=None, overlap=False):
        """Compute the matrix element <atom1, l1|H|l2, atom2>.
        The function is called in the member function initialize() and invokes the function
        me() from the module diatomic_matrix_element.

        Parameters
        ----------
        atom1 : int
            Atom index
        atom2 : int
            Atom index
        l1 : int
            Index of a localized basis function
        l2 : int
            Index of a localized basis function
        coords : numpy.ndarray
            Coordinates of radius vector pointing from one atom to another
            it may differ from the actual coordinates of atoms (Default value = None)
        overlap : bool
            A flag indicating that the overlap matrix element has to be computed

        Returns
        -------
        type float
            Inter-cites matrix element
        """

        # on site (pick right table of parameters for a certain atom)
        if atom1 == atom2 and coords is None:
            atom_obj = self._ind2atom(atom1)
            if l1 == l2:
                if overlap:
                    return 1.0
                else:
                    sym = 'epsilon_' + atom_obj.orbitals[l1]['title'] + '^' + atom_obj.title
                    return Symbol(sym)
            else:
                return self._comp_so(atom_obj, l1, l2)

        # nearest neighbours (define bound type and atomic quantum numbers)
        if atom1 != atom2 or coords is not None:

            atom_kind1 = self._ind2atom(atom1)
            atom_kind2 = self._ind2atom(atom2)

            # compute radius vector pointing from one atom to another
            if coords is None:
                coords1 = np.array(list(self.atom_list.values())[atom1], dtype=float) - \
                          np.array(list(self.atom_list.values())[atom2], dtype=float)
            else:
                coords1 = coords.copy()

            norm = np.linalg.norm(coords1)

            if verbosity.VERBOSITY > 0:

                coordinates = np.array2string(norm, precision=4) + " Ang between atoms " + \
                              self._ind2atom(atom1).title + " and " + self._ind2atom(atom2).title

                if coordinates not in unique_distances:
                    unique_distances.add(coordinates)

            if self.int_radial_dependence is None:
                which_neighbour = ""
            else:
                which_neighbour = self.int_radial_dependence(norm)

            if self.radial_dependence is None:
                factor = 1.0
            else:
                factor = self.radial_dependence(norm)

            # compute directional cosines
            if self.compute_angular:
                coords1 /= norm
            else:
                coords1 = np.array([1.0, 0.0, 0.0])

            ans = me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour,
                      overlap=overlap) * factor
            return ans

    def _reset_periodic_bc(self):
        """Reset the matrices determining periodic boundary conditions to their default state
        :return:

        Parameters
        ----------

        Returns
        -------

        """

        self.h_matrix_bc_add = zeros(self.basis_size, self.basis_size)
        self.ov_matrix_bc_add = zeros(self.basis_size, self.basis_size)
        self.h_matrix_bc_factor = ones(self.basis_size, self.basis_size)
        self.k_vector = None

    def _compute_phase(self, coords):
        angle = to_symbol(np.dot(self.k_vector, coords))
        phase = exp(I * angle)
        return phase

    def _get_tb_matrix(self, k_vector=None):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        Parameters
        ----------
        k_vector : numpy.ndarray
            wave vector

        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """

        if k_vector is None:
            return nsimplify(simplify(self.h_matrix))
        else:
            k_vector = list(k_vector)

            # reset previous wave vector if any
            if k_vector != self.k_vector:
                self._reset_periodic_bc()
                self.k_vector = k_vector
                self._compute_h_matrix_bc_factor()
                self._compute_h_matrix_bc_add(overlap=self.compute_overlap)

            ans = matrix_multiply_elementwise(self.h_matrix_bc_factor, self.h_matrix) + self.h_matrix_bc_add

            if self.compute_overlap:
                ans1 = matrix_multiply_elementwise(self.h_matrix_bc_factor, self.ov_matrix) + self.ov_matrix_bc_add
                return ans, ans1
            else:
                return ans

    def get_tb_matrix(self, k_vector=None):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        Parameters
        ----------
        k_vector : numpy.ndarray
            wave vector

        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """

        ans = self._get_tb_matrix(k_vector=k_vector)

        if isinstance(ans, tuple):
            ans1 = nsimplify(simplify(ans[0]))
            ans2 = nsimplify(simplify(ans[1]))
            return ans1, ans2
        else:
            ans = nsimplify(simplify(ans))
            return ans

    def get_tb_element(self, j1, j2, k_vector=None):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        Parameters
        ----------
        k_vector : numpy.ndarray
            wave vector

        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """

        ans = self._get_tb_matrix(k_vector=k_vector)

        if isinstance(ans, tuple):
            ans1 = nsimplify(simplify(ans[0][j1, j2]))
            ans2 = nsimplify(simplify(ans[1][j1, j2]))
            return ans1, ans2
        else:
            ans1 = simplify(nsimplify(simplify(ans[j1, j2])))
            return ans1

    def get_tb_matrix_latex(self, k_vector):

        ans = self.get_tb_matrix(k_vector)

        if isinstance(ans, tuple):
            return latex(ans[0]), latex(ans[1])
        else:
            return latex(ans)

    def get_eigenvalues(self, k_vector=None):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        Parameters
        ----------
        k_vector : numpy.ndarray
            wave vector

        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """

        ans = self.get_tb_matrix(k_vector=k_vector)

        # reset previous wave vector if any
        if isinstance(ans, tuple):
            return ans[0].eigenvals()
        else:
            return ans.eigenvals()

    def get_hamiltonians(self):
        """Return a list of Hamiltonian matrices. For 1D systems, the list is [Hl, Hc, Hr],
        where Hc is the Hamiltonian describing interactions between atoms within a unit cell,
        Hl and Hr are Hamiltonians describing couplings between atoms in the unit cell
        and atoms in the left and right adjacent unit cells.

        Parameters
        ----------

        Returns
        -------
        list
            list of Hamiltonians

        """

        self.k_vector = [0.0, 0.0, 0.0]

        self.h_matrix_left_lead = zeros(self.basis_size, self.basis_size)
        self.h_matrix_right_lead = zeros(self.basis_size, self.basis_size)

        self._compute_h_matrix_bc_add(split_the_leads=True)
        self.k_vector = None

        return nsimplify(simplify(self.h_matrix_left_lead.T)),\
               nsimplify(simplify(self.h_matrix)),\
               nsimplify(simplify(self.h_matrix_right_lead.T))

    def diagonalize_periodic_bc(self, k_vector):
        raise AttributeError( "diagonalize_periodic_bc() is not implemented for symbolic computations" )

    def diagonalize(self):
        raise AttributeError( "diagonalize() is not implemented for symbolic computations" )


def main1():
    from hamiltonian_initializer import set_tb_params

    a = Orbitals('A')
    a.add_orbital('s', -0.7)
    b = Orbitals('B')
    b.add_orbital('s', -0.5)
    c = Orbitals('C')
    c.add_orbital('s', -0.3)

    Orbitals.orbital_sets = {'A': a, 'B': b, 'C': c}

    set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
                     PARAMS_B_B={'ss_sigma': -0.5},
                     PARAMS_A_B={'ss_sigma': -0.5},
                     PARAMS_B_C={'ss_sigma': -0.5},
                     PARAMS_A_C={'ss_sigma': -0.5})

    xyz_file = """4
    H cell
    A1       0.0000000000    0.0000000000    0.0000000000
    B2       0.0000000000    0.0000000000    1.0000000000
    A2       0.0000000000    1.0000000000    0.0000000000
    B3       0.0000000000    1.0000000000    1.0000000000
    """

    h = HamiltonianSym(xyz=xyz_file, nn_distance=1.1)
    h.initialize()
    per = 2.0
    h.set_periodic_bc([[0, 0, per]])
    # h_l, h_0, h_r = h.get_hamiltonians()
    #
    # energy = np.linspace(-3.0, 1.5, 700)

    wv = np.linspace(0, 2*np.pi/per, 50)
    hl, h0, hr = h.get_hamiltonians()

    for item in wv:

        M = h.get_tb_matrix(np.array([0, 0, item]))
        pprint(M)
        # pprint(M.eigenvals())

    print('hi')


def main():
    from hamiltonian_initializer import set_tb_params

    a = Orbitals('A')
    a.add_orbital('s', -0.7)

    Orbitals.orbital_sets = {'A': a}

    set_tb_params(PARAMS_A_A={'ss_sigma': -0.5})

    xyz_file = """1
    A cell
    A1       0.0000000000    0.0000000000    0.0000000000
    """

    h = HamiltonianSym(xyz=xyz_file, nn_distance=1.1)
    h.initialize()
    per = 1.0
    h.set_periodic_bc([[0, 0, per]])
    wv = np.linspace(0, 2*np.pi/per, 50)

    for item in wv:
        M = h.get_tb_matrix(np.array([0, 0, item]))
        # M = h.get_tb_matrix(Symbol('kx'))
        pprint(M)
        # pprint(M.eigenvals())

    print('hi')


if __name__=='__main__':

    main1()
