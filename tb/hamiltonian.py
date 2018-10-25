"""
The module contains all necessary classes needed to compute the Hamiltonian matrix
"""
from __future__ import print_function, division
from __future__ import absolute_import
from collections import OrderedDict
from operator import mul
import matplotlib.pyplot as plt
import numpy as np
from tb.abstract_interfaces import AbstractBasis
from tb.structure_designer import StructDesignerXYZ, CyclicTopology
from tb.diatomic_matrix_element import me
from tb.atoms import Atom
from tb.aux_functions import dict2xyz
from tb.tb_script import postprocess_data
from functools import reduce


VERBOSITY = 1


class BasisTB(AbstractBasis, StructDesignerXYZ):
    """
    The class contains information about sets of quantum numbers and
    dimensionality of the Hilbert space.
    It is also equipped with the member functions translating quantum numbers
    into a raw index and vise versa.
    """

    def __init__(self, xyz, nn_distance):

        # parent class StructDesignerXYZ stores atom list initialized from xyz-file
        super(BasisTB, self).__init__(xyz=xyz, nn_distance=nn_distance)

        # each entry of the dictionary stores a label of the atom species as a key and
        # corresponding Atom object as a value. Each atom object contains infomation about number,
        # energy and symmetry of the orbitals
        self._orbitals_dict = Atom.atoms_factory(list(self.num_of_species.keys()))

        # `quantum_number_lims` counts number of species and corresponding number
        # of orbitals for each; each atom kind is enumerated
        self.quantum_numbers_lims = []
        for item in list(self.num_of_species.keys()):
            self.quantum_numbers_lims.append(OrderedDict([('atoms', self.num_of_species[item]),
                                                          ('l', self.orbitals_dict[item].num_of_orbitals)]))

        # count total number of basis functions
        self.basis_size = 0
        for item in self.quantum_numbers_lims:
            self.basis_size += reduce(mul, list(item.values()))

        # compute offset index for each atom
        self._offsets = [0]
        for j in range(len(self.atom_list)-1):
            self._offsets.append(self.orbitals_dict[list(self.atom_list.keys())[j]].num_of_orbitals)
        self._offsets = np.cumsum(self._offsets)

    def qn2ind(self, qn):

        qn = OrderedDict(qn)

        if list(qn.keys()) == list(self.quantum_numbers_lims[0].keys()):  # check if the input is
                                                              # a proper set of quantum numbers
            return self._offsets[qn['atoms']] + qn['l']
        else:
            raise IndexError("Wrong set of quantum numbers")

    def ind2qn(self, ind):
        pass  # TODO

    @property
    def orbitals_dict(self):

        class MyDict(dict):
            def __getitem__(self, key):
                key = ''.join([i for i in key if not i.isdigit()])
                return super(MyDict, self).__getitem__(key)

        return MyDict(self._orbitals_dict)


class Hamiltonian(BasisTB):
    """
    Class defines a Hamiltonian matrix as well as a set of member-functions
    allowing to build, diagonalize and visualize the matrix.
    """

    def __init__(self, **kwargs):

        xyz = kwargs.get('xyz', "")
        nn_distance = kwargs.get('nn_distance', 2.39)

        if isinstance(xyz, str):
            super(Hamiltonian, self).__init__(xyz=xyz, nn_distance=nn_distance)
        else:
            super(Hamiltonian, self).__init__(xyz=dict2xyz(xyz), nn_distance=nn_distance)

        self._coords = None                               # coordinates of sites
        self.h_matrix = None                            # Hamiltonian for an isolated system
        self.h_matrix_bc_factor = None                  # exponential Bloch factors for pbc
        self.h_matrix_bc_add = None                     # additive Bloch exponentials for pbc
                                                        # (interaction with virtual neighbours
                                                        # in adacent primitive cells due to pbc)

        self.h_matrix_left_lead = None
        self.h_matrix_right_lead = None
        self.k_vector = 0                               # default value of the wave vector
        self.ct = None
        self.radial_dependence = None

    def initialize(self, radial_dep=None):
        """
        The function computes matrix elements of the Hamiltonian.
        """

        self.radial_dependence = radial_dep
        self._coords = [0 for _ in range(self.basis_size)]
        # initialize Hamiltonian matrices
        self.h_matrix = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)

        # loop over all nodes
        for j1 in range(self.num_of_nodes):

            # find neighbours for each node
            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                # on site interactions
                if j1 == j2:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1, radial_dep=self.radial_dependence)
                        self._coords[ind1] = list(self.atom_list.values())[j1]

                # nearest neighbours interaction
                else:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2, radial_dep=self.radial_dependence)

    def set_periodic_bc(self, primitive_cell, radial_dep):

        if list(primitive_cell):
            self.ct = CyclicTopology(primitive_cell, list(self.atom_list.keys()), list(self.atom_list.values()), self._nn_distance)

            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            coordinates_to_plot = np.asarray(list(self.atom_list.values()))
            ax.scatter(coordinates_to_plot[:, 0], coordinates_to_plot[:, 1], coordinates_to_plot[:, 2], c='red', s=100)

            map1 = [item.startswith('*_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]
            map2 = [item.startswith('**_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]

            coordinates_to_plot = np.asarray(list(self.ct.virtual_and_interfacial_atoms.values()))
            ax.scatter(coordinates_to_plot[map1, 0], coordinates_to_plot[map1, 1], coordinates_to_plot[map1, 2],
                       c='green', s=70)
            ax.scatter(coordinates_to_plot[map2, 0], coordinates_to_plot[map2, 1], coordinates_to_plot[map2, 2],
                       s=20)

            plt.show()

            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            coordinates_of_atoms_in_unit_cell = np.asarray(list(self.atom_list.values()))
            ax.scatter(coordinates_of_atoms_in_unit_cell[:, 0], coordinates_of_atoms_in_unit_cell[:, 1], coordinates_of_atoms_in_unit_cell[:, 2], c='k', s=60)

            coordinates_of_atoms_outside_of_unit_cell = np.asarray(list(self.ct.virtual_and_interfacial_atoms.values()))
            for ii in range(len(self.atom_list)):
                for jj in range(len(self.ct.virtual_and_interfacial_atoms)):
                    if radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 1:
                       ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                                  coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                                  coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                                  c='r', s=30)
                    elif radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 2:
                        ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                                   coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                                   coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                                   c='g', s=20)
                    elif radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 3:
                        ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                                   coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                                   coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                                   c='b', s=10)

            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            plt.show()

        else:
            self.ct = None

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian matrix for the finite isolated system
        :return:
        """

        vals, vects = np.linalg.eigh(self.h_matrix)
        vals = np.real(vals)
        ind = np.argsort(vals)
        return vals[ind], vects[:, ind]

    def diagonalize_periodic_bc(self, k_vector):
        """
        Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        :param k_vector:   wave vector
        :return:
        """

        k_vector = list(k_vector)

        # reset previous wave vector if any
        if k_vector != self.k_vector:

            self._reset_periodic_bc()
            self.k_vector = k_vector
            self._compute_h_matrix_bc_factor()
            self._compute_h_matrix_bc_add()

        vals, vects = np.linalg.eigh(self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add)
        vals = np.real(vals)
        ind = np.argsort(vals)

        return vals[ind], vects[:, ind]

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

        return True

    def _get_me(self, atom1, atom2, l1, l2, coords=None, radial_dep=None, spin_orbit=None):
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
        if atom1 == atom2 and l1 == l2 and coords is None:
            return self.orbitals_dict[list(self.atom_list.keys())[atom1]].orbitals[l1]['energy']

        # nearest neighbours (define bound type and atomic quantum numbers)
        if atom1 != atom2 or coords is not None:

            atom_kind1 = self.orbitals_dict[list(self.atom_list.keys())[atom1]]
            atom_kind2 = self.orbitals_dict[list(self.atom_list.keys())[atom2]]

            # compute radius vector pointing from one atom to another
            if coords is None:
                coords1 = np.array(list(self.atom_list.values())[atom1], dtype=float) - \
                         np.array(list(self.atom_list.values())[atom2], dtype=float)
            else:
                coords1 = coords.copy()

            if radial_dep is None:
                which_neighbour = ""
            else:
                which_neighbour = radial_dep(coords1)

            # compute directional cosines
            coords1 /= np.linalg.norm(coords1)

            if VERBOSITY > 1:
                print("coords = ", coords1)
                print(list(self.atom_list.values())[atom1])
                print(list(self.atom_list.keys())[atom1])
                print(list(self.atom_list.values())[atom2])
                print(list(self.atom_list.keys())[atom2])
                print(atom_kind1.title, atom_kind2.title)

            return me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour)

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

        for j1 in range(self.num_of_nodes):

            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                if j1 != j2:
                    coords = np.array(list(self.atom_list.values())[j1], dtype=float) - \
                             np.array(list(self.atom_list.values())[j2], dtype=float)
                    phase = np.exp(1j * np.dot(self.k_vector, coords))

                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):

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

            list_of_neighbours = self.ct.get_neighbours(list(self.atom_list.values())[j1])

            for j2 in list_of_neighbours:

                coords = np.array(list(self.atom_list.values())[j1]) - \
                         np.array(list(self.ct.virtual_and_interfacial_atoms.values())[j2])

                if split_the_leads:
                    flag = self.ct.atom_classifier(list(self.ct.virtual_and_interfacial_atoms.values())[j2], self.ct.pcv[0])

                phase = np.exp(1j*np.dot(self.k_vector, coords))

                ind = int(list(self.ct.virtual_and_interfacial_atoms.keys())[j2].split('_')[2])

                for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                    for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[ind]].num_of_orbitals):

                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                        ind2 = self.qn2ind([('atoms', ind), ('l', l2)])

                        if split_the_leads:
                            if flag == 'R':
                                self.h_matrix_left_lead[ind1, ind2] += phase * \
                                    self._get_me(j1, ind, l1, l2, coords, radial_dep=self.radial_dependence)
                            elif flag == 'L':
                                self.h_matrix_right_lead[ind1, ind2] += phase * \
                                    self._get_me(j1, ind, l1, l2, coords, radial_dep=self.radial_dependence)
                            else:
                                raise ValueError("Wrong flag value")
                        else:
                            self.h_matrix_bc_add[ind1, ind2] += phase * \
                                self._get_me(j1, ind, l1, l2, coords, radial_dep=self.radial_dependence)

    def get_coupling_hamiltonians(self):

        self.k_vector = [0.0, 0.0, 0.0]
        self._compute_h_matrix_bc_add(split_the_leads=True)
        return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T

    def get_site_coordinates(self):
        """
        Returns coordinates of atoms in the order of Hamiltonian matrix indexing

        :return:
        """

        return np.array(self._coords)
