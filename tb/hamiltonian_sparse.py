"""
The module contains all necessary classes needed to compute the Hamiltonian matrix
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
import constants
from atoms import Atom
from hamiltonian import Hamiltonian


class HamiltonianSp(Hamiltonian):
    """
    Class defines a Hamiltonian matrix as well as a set of member-functions
    allowing to build, diagonalize and visualize the matrix.
    """

    def __init__(self, **kwargs):

        super(HamiltonianSp, self).__init__(**kwargs)

    def initialize(self):
        """
        The function computes matrix elements of the Hamiltonian.
        """

        # initialize Hamiltonian matrices
        self.h_matrix = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_add = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = sp.lil_matrix(self.h_matrix_bc_factor)

        # loop over all nodes
        for j1 in xrange(self.num_of_nodes):

            # find neighbours for each node
            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                # on site interactions
                if j1 == j2:
                    for l1 in xrange(self.orbitals_dict[self.atom_list.keys()[j1]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1)

                # nearest neighbours interaction
                else:
                    for l1 in xrange(self.orbitals_dict[self.atom_list.keys()[j1]].num_of_orbitals):
                        for l2 in xrange(self.orbitals_dict[self.atom_list.keys()[j2]].num_of_orbitals):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian matrix for the finite isolated system
        :return:
        """

        vals, vects = splin.eigs(self.h_matrix)
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

        vals, vects = splin.eigs(self.h_matrix_bc_factor.multiply(self.h_matrix) + self.h_matrix_bc_add,
                                 k=10, sigma=2.1)
        vals = np.real(vals)
        ind = np.argsort(vals)

        return vals[ind], vects[:, ind]

    def _reset_periodic_bc(self):
        """
        Resets the matrices determining periodic boundary conditions to their default state
        :return:
        """

        self.h_matrix_bc_add = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = sp.lil_matrix(self.h_matrix_bc_factor)
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

                    for l1 in xrange(self.orbitals_dict[self.atom_list.keys()[j1]].num_of_orbitals):
                        for l2 in xrange(self.orbitals_dict[self.atom_list.keys()[j2]].num_of_orbitals):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix_bc_factor[ind1, ind2] = phase
                            # self.h_matrix[ind2, ind1] = self.h_matrix[ind1, ind2]

    def _compute_h_matrix_bc_add(self, split_the_leads=False):
        """
            Compute additive Bloch exponentials needed to specify pbc
        """

        if split_the_leads:
            self.h_matrix_left_lead = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
            self.h_matrix_right_lead = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
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

                for l1 in xrange(self.orbitals_dict[self.atom_list.keys()[j1]].num_of_orbitals):
                    for l2 in xrange(self.orbitals_dict[self.atom_list.keys()[ind]].num_of_orbitals):

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


def main():

    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]
    Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    h = HamiltonianSp(xyz='/home/mk/TB_project/input_samples/SiNW.xyz')
    h.initialize()
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 30
    kk = np.linspace(0, constants.PI / a_si, num_points, endpoint=True)
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

    split = 100
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylim(-1.0, -0.3)
    ax[0].plot(kk, np.sort(np.real(band_sructure))[:, :split])
    ax[0].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[0].set_ylabel(r'Energy (eV)')
    ax[0].set_title('Valence band')

    ax[1].set_ylim(2.0, 2.7)
    ax[1].plot(kk, np.sort(np.real(band_sructure))[:, split:])
    ax[1].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[1].set_ylabel(r'Energy (eV)')
    ax[1].set_title('Conduction band')
    fig.tight_layout()
    plt.savefig('test.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':

    main()
