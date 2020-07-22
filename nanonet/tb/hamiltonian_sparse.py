"""
The module contains all necessary classes needed to compute the Hamiltonian matrix
"""
from __future__ import print_function, division
from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
from nanonet.tb.orbitals import Orbitals
from nanonet.tb.hamiltonian import Hamiltonian


class HamiltonianSp(Hamiltonian):
    """Class defines a Hamiltonian matrix as well as a set of member-functions
    allowing to build, diagonalize and visualize the matrix.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, **kwargs):

        super(HamiltonianSp, self).__init__(**kwargs)

        sigma = kwargs.get('sigma', 1.1)
        num_eigs = kwargs.get('num_eigs', 14)

        self.sigma = sigma
        self.num_eigs = num_eigs

    def initialize(self):
        """The function computes matrix elements of the Hamiltonian."""

        # initialize Hamiltonian matrices
        self.h_matrix = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_add = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = sp.lil_matrix(self.h_matrix_bc_factor)

        # loop over all nodes
        for j1 in range(self.num_of_nodes):

            # find neighbours for each node
            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                # on site interactions
                if j1 == j2:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1)

                # nearest neighbours interaction
                else:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)

        return self

    def diagonalize(self):
        """Diagonalize the Hamiltonian matrix for the finite isolated system
        :return:

        Parameters
        ----------

        Returns
        -------

        """

        vals, vects = splin.eigsh(self.h_matrix, k=self.num_eigs, sigma=self.sigma)
        vals = np.real(vals)
        ind = np.argsort(vals)
        return vals[ind], vects[:, ind]

    def diagonalize_periodic_bc(self, k_vector):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector

        Parameters
        ----------
        k_vector :
            wave vector

        Returns
        -------

        """

        k_vector = list(k_vector)

        # reset previous wave vector if any
        if k_vector != self.k_vector:

            self._reset_periodic_bc()
            self.k_vector = k_vector
            self._compute_h_matrix_bc_factor()
            self._compute_h_matrix_bc_add()

        try:
            vals, vects = splin.eigsh(self.h_matrix_bc_factor.multiply(self.h_matrix) + self.h_matrix_bc_add,
                                      k=self.num_eigs, sigma=self.sigma)
        except TypeError:
            mat = self.h_matrix_bc_factor.multiply(self.h_matrix) + self.h_matrix_bc_add
            vals, vects = np.linalg.eigh(mat.todense())

        vals = np.real(vals)
        ind = np.argsort(vals)

        return vals[ind], vects[:, ind]

    def _reset_periodic_bc(self):
        """Resets the matrices determining periodic boundary conditions to their default state
        :return:

        Parameters
        ----------

        Returns
        -------

        """

        self.h_matrix_bc_add = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = np.ones((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_bc_factor = sp.lil_matrix(self.h_matrix_bc_factor)
        self.k_vector = None

    def get_hamiltonians(self):
        """ """

        self.k_vector = [0.0, 0.0, 0.0]

        self.h_matrix_left_lead = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)
        self.h_matrix_right_lead = sp.lil_matrix((self.basis_size, self.basis_size), dtype=np.complex)

        self._compute_h_matrix_bc_add(split_the_leads=True)
        self.k_vector = None

        return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T


def main():
    """ """

    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    h = HamiltonianSp(xyz='/home/mk/TB_project/input_samples/SiNW.xyz')
    h.initialize()
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 30
    kk = np.linspace(0, np.PI / a_si, num_points, endpoint=True)
    band_sructure = []

    for jj in range(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_sructure.append(vals)

    band_sructure = np.array(band_sructure)

    vb = np.sort(np.real(band_sructure))
    cb = vb.copy()
    vb[vb > 0] = np.NaN
    cb[cb < 0] = np.NaN

    vb = -np.sort(-vb)
    cb = np.sort(cb)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylim(-1.0, -0.3)
    ax[0].plot(kk, vb)
    ax[0].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[0].set_ylabel(r'Energy (eV)')
    ax[0].set_title('Valence band')

    ax[1].set_ylim(2.0, 2.7)
    ax[1].plot(kk, cb)
    ax[1].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[1].set_ylabel(r'Energy (eV)')
    ax[1].set_title('Conduction band')
    fig.tight_layout()
    plt.savefig('test.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':

    main()
