"""
The module contains all necessary classes needed to compute th Hamiltonian matrix
"""
from collections import OrderedDict
from operator import mul
import matplotlib.pyplot as plt
import numpy as np
from abstract_interfaces import AbstractBasis
from structure_designer import StructDesignerXYZ, CyclicTopology
from params import *
from diatomic_matrix_element import me


class Basis(AbstractBasis):
    """
    The class contains information about sets of quantum numbers and
    dimensionality of the Hilbert space.
    It is also equipped with the member functions translating quantum numbers
    into a raw index and vise versa.
    """

    def __init__(self, *args):

        if args is None:
            self.quantum_numbers_lims = [OrderedDict([('i', 2)])]
        else:

            if len(args) == 1:
                self.quantum_numbers_lims = [OrderedDict(args[0])]
            else:
                self.quantum_numbers_lims = []
                for item in args:
                    self.quantum_numbers_lims.append(OrderedDict(item))

        self.basis_size_subsystem = []
        for item in self.quantum_numbers_lims:
            self.basis_size_subsystem.append(reduce(mul, item.values()))

        self.basis_size = sum(self.basis_size_subsystem)

    def qn2ind(self, qn, subsystem=0):

        if subsystem > len(self.quantum_numbers_lims)-1:
            raise ValueError('The number of subsystems excess the declared one')

        qn = OrderedDict(qn)

        if subsystem > 0:
            offset = sum(self.basis_size_subsystem[:subsystem])
        else:
            offset = 0

        # check if the input is a proper set of quantum numbers
        if qn.keys() == self.quantum_numbers_lims[subsystem].keys():
            return np.ravel_multi_index(qn.values(),
                                        self.quantum_numbers_lims[subsystem].values()) + offset
        else:
            raise IndexError("Wrong set of quantum numbers")

    def ind2qn(self, ind):

        offsets = np.cumsum(self.basis_size_subsystem)

        subsystem = 0

        while subsystem < len(self.basis_size_subsystem)-1:

            if ind > offsets[subsystem+1] and ind >= offsets[subsystem]:
                break

            subsystem += 1

        if subsystem > 0:
            offsets = offsets[subsystem-1]
        else:
            offsets = 0

        indices = np.unravel_index(ind-offsets, self.quantum_numbers_lims[subsystem].values())
        qn = OrderedDict(zip(self.quantum_numbers_lims[subsystem].keys(), indices))
        return qn, subsystem


class BasisTB(AbstractBasis, StructDesignerXYZ):
    """
    The class contains information about sets of quantum numbers and
    dimensionality of the Hilbert space.
    It is also equipped with the member functions translating quantum numbers
    into a raw index and vise versa.
    """

    num_of_orbitals = {'S': 10, 'H': 1}

    def __init__(self, xyz):

        super(BasisTB, self).__init__(xyz=xyz)

        self.quantum_numbers_lims = []

        for item in self.num_of_species.keys():
            self.quantum_numbers_lims.append(OrderedDict([('atoms', self.num_of_species[item]),
                                                          ('l', BasisTB.num_of_orbitals[item])]))

        self.basis_size_subsystem = []
        for item in self.quantum_numbers_lims:
            self.basis_size_subsystem.append(reduce(mul, item.values()))

        self.basis_size = sum(self.basis_size_subsystem)

        self._offsets = [0]

        keys = self.atom_list.keys()

        for j in xrange(len(keys)-1):

            self._offsets.append(BasisTB.num_of_orbitals[keys[j][0]])

        self._offsets = np.cumsum(self._offsets)

    def qn2ind(self, qn):

        qn = OrderedDict(qn)

        if qn.keys() == self.quantum_numbers_lims[0].keys():  # check if the input is
                                                              # a proper set of quantum numbers
            return self._offsets[qn['atoms']] + qn['l']
        else:
            raise IndexError("Wrong set of quantum numbers")

    def ind2qn(self, ind):
        pass  # TODO


class Hamiltonian(BasisTB):
    """
    Class defines a Hamiltonian matrix as well as a set of member-functions
    allowing to build, diagonalize and visualize the matrix.
    """

    def __init__(self, **kwargs):

        xyz = kwargs.get('xyz', "")

        super(Hamiltonian, self).__init__(xyz=xyz)
        self.h_matrix = None                            # Hamiltonian for an isolated system
        self.h_matrix_bc_factor = None                  # exponential Bloch factors for pbc
        self.h_matrix_bc_add = None                     # additive Bloch exponentials for pbc
                                                        # (interaction with virtual neighbours
                                                        # in adacent primitive cells due to pbc)

        self.h_matrix_left_lead = None
        self.h_matrix_right_lead = None
        self.k_vector = 0                               # default value of the wave vector
        self.ct = None

    def initialize(self):
        """
        The function computes matrix elements of the Hamiltonian.
        """

        # initialize Hamiltonian matrices
        self.h_matrix = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)

        # loop over all nodes
        for j1 in xrange(self.num_of_nodes):

            # find neighbours for each node
            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                # on site interactions
                if j1 == j2:
                    for l1 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j1][0]]):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1)

                # nearest neighbours interaction
                else:
                    for l1 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j1][0]]):
                        for l2 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j2][0]]):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)

    def set_periodic_bc(self, primitive_cell):

        if len(primitive_cell) > 0:
            self.ct = CyclicTopology(primitive_cell, self.atom_list.keys(), self.atom_list.values())
        else:
            self.ct = None

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian matrix for the finite isolated system
        :return:
        """

        if len(self.k_vector) != 0:
            self._reset_periodic_bc()

        return np.linalg.eig(self.h_matrix)

    def diagonalize_periodic_bc(self, k_vector):
        """
        Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        :param k_vector:   wave vector
        :return:
        """

        # reset previous wave vector if any
        if k_vector != self.k_vector:

            self._reset_periodic_bc()
            self.k_vector = k_vector
            self._compute_h_matrix_bc_factor()
            self._compute_h_matrix_bc_add()

        return np.linalg.eig(self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add)

    def is_hermitian(self, visualize=False):
        """
        Check if the Hamiltonian is a Hermitian matrix. Useful for testing.

        :param visualize:   Show the matrix with the non-Hermitian elements marked
        :return:            True is the Hamiltonian is a Hermitian matrix, False otherwise
        """

        h_matrix = self.h_matrix * self.h_matrix_bc_factor + self.h_matrix_bc_add

        if visualize:
            plt.imshow((np.abs(h_matrix - h_matrix.conj().T)))

        if (np.abs(h_matrix - h_matrix.conj().T) > 0.001).any():
            return False
        else:
            return True

    def _get_me(self, atom1, atom2, l1, l2, coords=None):
        """
        Compute the matrix element <atom1, l1|H|l2, atom2>

        :param atom1:    atom index
        :param atom2:    atom index
        :param l1:       index of a localized basis function
        :param l2:       index of a localized basis function
        :param coords:   imposed coordinates of radius vector pointing from one atom to another
                         it may differ from the actual coordinates of atoms
        :return:         matrix element
        :rtype:          float
        """

        # on site (pick right table of parameters for a certain atom)
        if atom1 == atom2 and self.atom_list.keys()[atom1].startswith("Si") and l1 == l2:
            return PARAMS_ON_SITE_SI[ORBITALS_CODE[l1][0]]

        if atom1 == atom2 and self.atom_list.keys()[atom1].startswith("H") and l1 == l2:
            return PARAMS_ON_SITE_H[ORBITALS_CODE[l1][0]]

        # nearest neighbours (define bound type and atomic quantum numbers)
        if atom1 != atom2:
            if self.atom_list.keys()[atom1].startswith("Si") and \
                    self.atom_list.keys()[atom2].startswith("Si"):
                neighbours = "Si-Si"
            else:
                neighbours = "Si-H"

            # quantum numbers for the first atom
            n1 = ORBITALS_CODE_N[l1]
            ll1 = ORBITALS_CODE_L[l1]
            m1 = ORBITALS_CODE_M[l1]

            # quantum numbers for the second atom
            n2 = ORBITALS_CODE_N[l2]
            ll2 = ORBITALS_CODE_L[l2]
            m2 = ORBITALS_CODE_M[l2]

            # compute radius vector pointing from one atom to another
            if coords is None:
                coords = np.array(self.atom_list.values()[atom1], dtype=float) - \
                         np.array(self.atom_list.values()[atom2], dtype=float)

            # compute directional cosines
            coords /= np.linalg.norm(coords)

            if VERBOSITY > 1:
                print "coords = ", coords
                print self.atom_list.values()[atom1]
                print self.atom_list.keys()[atom1]
                print self.atom_list.values()[atom2]
                print self.atom_list.keys()[atom2]

            return me(neighbours, n1, ll1, m1, n2, ll2, m2, coords)

    def _reset_periodic_bc(self):
        """
        Resets the matrices determining periodic boundary conditions to their default state
        :return:
        """

        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)
        self.k_vector = None

    def _compute_h_matrix_bc_factor(self):
        """
        Compute the exponential Bloch factors needed to specify pbc
        """

        for j1 in xrange(self.num_of_nodes):

            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                if j1 != j2:
                    coords = np.array(self.atom_list.values()[j1], dtype=float) - \
                             np.array(self.atom_list.values()[j2], dtype=float)
                    phase = np.exp(1j * np.dot(self.k_vector, coords))

                    for l1 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j1][0]]):
                        for l2 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j2][0]]):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix_bc_factor[ind1, ind2] = phase
                            # self.h_matrix[ind2, ind1] = self.h_matrix[ind1, ind2]

    def _compute_h_matrix_bc_add(self, split_the_leads=False):
        """
            Compute additive Bloch exponentials needed to specify pbc
        """

        if split_the_leads:
            self.h_matrix_left_lead = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
            self.h_matrix_right_lead = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
            flag = None

        # loop through all interfacial atoms
        for j1 in self.ct.interfacial_atoms_ind:

            list_of_neighbours = self.ct.get_neighbours(self.atom_list.values()[j1])

            for j2 in list_of_neighbours:

                coords = np.array(self.atom_list.values()[j1]) - \
                         np.array(self.ct.virtual_and_interfacial_atoms.values()[j2])

                if split_the_leads:
                    flag = self.ct.atom_classifier(self.ct.virtual_and_interfacial_atoms.values()[j2], self.ct.pcv[0])

                phase = np.exp(1j*np.dot(self.k_vector, coords))

                ind = int(self.ct.virtual_and_interfacial_atoms.keys()[j2].split('_')[2])

                for l1 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[j1][0]]):
                    for l2 in xrange(BasisTB.num_of_orbitals[self.atom_list.keys()[ind][0]]):

                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                        ind2 = self.qn2ind([('atoms', ind), ('l', l2)])

                        if split_the_leads:
                            if flag == 'R':
                                self.h_matrix_left_lead[ind1, ind2] += phase * \
                                    self._get_me(j1, ind, l1, l2, coords)
                            elif flag == 'L':
                                self.h_matrix_right_lead[ind1, ind2] += phase * \
                                    self._get_me(j1, ind, l1, l2, coords)
                            else:
                                raise ValueError("Wrong flag value")
                        else:
                            self.h_matrix_bc_add[ind1, ind2] += phase * \
                                self._get_me(j1, ind, l1, l2, coords)

    def get_coupling_hamiltonians(self):

        self.k_vector = [0.0, 0.0, 0.0]
        self._compute_h_matrix_bc_add(split_the_leads=True)
        return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T


def format_func(value, tick_number):

    # if value == PI / a_si:
    #     return r"$\frac{\pi}{2}$"
    # else:
    #     return '%.2f' % value

    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def main():

    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]

    h = Hamiltonian(xyz='/home/mk/TB_project/tb/SiNW.xyz', primitive_cell=PRIMITIVE_CELL)
    h.initialize()
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 20
    kk = np.linspace(0, PI / a_si, num_points, endpoint=True)
    band_sructure = []

    for jj in xrange(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_sructure.append(vals)

    band_sructure = np.array(band_sructure)

    ax = plt.axes()
    ax.set_ylim(-1.0, 2.7)
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.plot(kk, np.sort(np.real(band_sructure)))
    plt.show()


def main1():

    from tb.aux_functions import get_k_coords

    # ----------------------------------------------------------------------

    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0.5 * a_si, 0.5 * a_si],
                      [0.5 * a_si, 0, 0.5 * a_si],
                      [0.5 * a_si, 0.5 * a_si, 0]]

    # ----------------------------------------------------------------------

    sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
    num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]

    # sym_points = ['L', 'GAMMA', 'X', 'K', 'GAMMA']
    # num_points = [15, 20, 5, 10]

    k = get_k_coords(sym_points, num_points)
    vals = np.zeros((sum(num_points), 20), dtype=np.complex)

    h = Hamiltonian(xyz='/home/mk/TB_project/tb/si.xyz',
                    primitive_cell=PRIMITIVE_CELL)
    h.initialize()
    h.set_periodic_bc(PRIMITIVE_CELL)

    for jj, i in enumerate(k):
        vals[jj, :], _ = h.diagonalize_periodic_bc(list(i))

    plt.plot(np.sort(np.real(vals)))
    plt.show()


if __name__ == '__main__':

    main()
