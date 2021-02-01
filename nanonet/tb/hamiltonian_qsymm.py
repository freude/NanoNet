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
from nanonet.tb.diatomic_matrix_element import me_qsymm
from nanonet.tb.orbitals import Orbitals
import nanonet.verbosity as verbosity
from nanonet.tb.hamiltonian import Hamiltonian


init_printing(use_unicode=True)


class HamiltonianQsymm(Hamiltonian):
    """Class defines a Hamiltonian matrix in the symbolic representation.
    """

    def __init__(self, **kwargs):
        super(HamiltonianQsymm, self).__init__(**kwargs)

    # def initialize(self, int_radial_dep=None, radial_dep=None):
    #     """Compute matrix elements of the Hamiltonian.
    #     Parameters
    #     ----------
    #     int_radial_dep : func
    #          Integer radial dependence function (Default value = None)
    #     radial_dep : func
    #          Radial dependence function (Default value = None)
    #     Returns
    #     -------
    #     type Hamiltonian
    #         Returns the instance of the class Hamiltonian
    #     """
    #
    #     super().initialize(int_radial_dep=int_radial_dep, radial_dep=radial_dep)

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
        print(atom1, atom2, l1, l2)
        # on site (pick right table of parameters for a certain atom)
        if atom1 == atom2 and coords is None:
            atom_obj = self._ind2atom(atom1)
            if l1 == l2:
                if overlap:
                    return 1.0
                else:
                    return atom_obj.orbitals[l1]['energy']
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

                coordinates = str(int(np.degrees(np.arccos(coords1[0]/norm)))) + u"\N{DEGREE SIGN} and " + \
                              np.array2string(norm, precision=4) + " Ang between atoms " + \
                              self._ind2atom(atom1).title + " and " + self._ind2atom(atom2).title

                if coordinates not in self.unique_distances:
                    self.unique_distances.add(coordinates)

            if self.int_radial_dependence is None:
                which_neighbour = ""
            else:
                which_neighbour = self.int_radial_dependence(norm)

            if self.radial_dependence is None:
                factor = 1.0
            else:
                factor = self.radial_dependence(norm)

            # compute directional cosines
            if not self.compute_angular:
                coords1 = np.array([1.0, 0.0, 0.0])

            return me_qsymm(atom_kind1, l1, atom_kind2, l2, coords1) * factor
