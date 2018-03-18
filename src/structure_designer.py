"""
The module contains classes serving to define
geometrical structure and boundary conditions of the problem.
"""

__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"
__version__ = "0.0.1"

from collections import OrderedDict
import numpy as np
import scipy.spatial
from src.third_party.cluster_100 import supercell
from src.aux_functions import xyz2np, count_species
from abstract_interfaces import AbstractStructureDesigner
from params import *


class StructDesignerXYZ(AbstractStructureDesigner):
    """
    The class builds the atomic structure from the xyz file or string.
    """

    def __init__(self, xyz='/home/mk/TB_project/src/my_si.xyz'):

        try:
            with open(xyz, 'rb') as read_file:
                reader = read_file.read()

        except IOError:
            reader = xyz

        labels, coords = xyz2np(reader)
        self._num_of_species = count_species(labels)
        self._num_of_nodes = sum(self.num_of_species.values())
        self._atom_list = OrderedDict(zip(labels, coords))
        self._kd_tree = scipy.spatial.cKDTree(coords, leafsize=100)

    @property
    def atom_list(self):
        return self._atom_list

    @property
    def num_of_nodes(self):
        return self._num_of_nodes

    @property
    def num_of_species(self):
        return self._num_of_species

    def get_neighbours(self, query):

        if isinstance(query, list):
            ans = self._kd_tree.query(query,
                                      k=5,
                                      distance_upper_bound=2.39)
        elif isinstance(query, int):
            ans = self._kd_tree.query(self.atom_list.items()[query][1],
                                      k=5,
                                      distance_upper_bound=2.39)
        elif isinstance(query, str):
            ans = self._kd_tree.query(self.atom_list[query],
                                      k=5,
                                      distance_upper_bound=2.39)
        else:
            raise TypeError('Wrong input type for query')

        ans1 = [ans[1][0]]

        for item in zip(ans[0], ans[1]):
            if 1.47 < item[0] < 2.39:
                ans1.append(item[1])

        return ans1


class StructDesigner(StructDesignerXYZ):
    """
    The class builds the atomic structure using
    a third party generating function supercell
    from the module cluster100.
    """

    def __init__(self):

        a_si = 5.50

        si = supercell(2, 2, 2, AFinite=True, BFinite=True, CFinite=False, RemoveSilane=False)
        si.scale(a_si)
        # si.translate(-A*aSi/2,-B*aSi/2,-C*aSi/2)
        # si.translate(0,0,-C*aSi/2)
        # si.translate(0, 0, 0)

        if si.closeatoms():
            print "Close contacts: ", si.closeatoms()
        a = si.toxyz()

        open('si.xyz', 'w').write(a)

        labels, coords = xyz2np(a)
        self._num_of_species = count_species(labels)
        self._num_of_nodes = sum(self.num_of_species.values())
        self._atom_list = OrderedDict(zip(labels, coords))
        self._kd_tree = scipy.spatial.cKDTree(coords, leafsize=100)


class CyclicTopology(object):
    """
    The class provides functionality for determining
    the periodic boundary conditions for a crystal cell.
    The object of the class is instantiated by
    a set of the primitive cell vectors.
    """

    def __init__(self, primitive_cell_vectors, labels, coords):

        self.pcv = primitive_cell_vectors

        # compute vectors' lengths
        self.sizes = []
        for item in self.pcv:
            self.sizes.append(np.linalg.norm(item))

        self.interfacial_atoms_ind = []
        self.virtual_and_interfacial_atoms = OrderedDict()
        self._generate_atom_list(labels, coords)

        self._kd_tree = scipy.spatial.cKDTree(self.virtual_and_interfacial_atoms.values(), leafsize=100)

    def belong_to_surfaces(self, coord):

        surfaces = []
        surfaces_adj = []

        thr1 = 0.1
        thr2 = thr1 + 1.37

        for j, item in enumerate(self.pcv):

            distance_to_surface = np.inner(coord, item) / self.sizes[j]
            distance_to_adj_surface = np.inner(coord - item, item) / self.sizes[j]

            if abs(distance_to_surface) < thr1:
                surfaces.append(j)

            if abs(distance_to_adj_surface) < thr2 and distance_to_adj_surface < 0:
                surfaces_adj.append(j)

        return surfaces, surfaces_adj

    def _generate_atom_list(self, labels, coords):

        count = 0

        for j, item in enumerate(coords):

            s_base, s_adj = self.belong_to_surfaces(item)

            if len(s_base) > 0:
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)

                for surf in s_base:
                    atom_coords = item + self.pcv[surf]
                    self.virtual_and_interfacial_atoms.update({"*_" + str(count) + "_" + str(j) + "_" + labels[j]: atom_coords})
                    count += 1

            if len(s_adj) > 0:
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)
                for surf in s_adj:
                    atom_coords = item - self.pcv[surf]
                    self.virtual_and_interfacial_atoms.update({"*_" + str(count) + "_" + str(j) + "_" + labels[j]: atom_coords})
                    count += 1

    def get_neighbours(self, query):

        if isinstance(query, list) or isinstance(query, np.ndarray):
            ans = self._kd_tree.query(query,
                                      k=5,
                                      distance_upper_bound=2.4)
        elif isinstance(query, int):
            ans = self._kd_tree.query(self.virtual_and_interfacial_atoms.items()[query][1],
                                      k=5,
                                      distance_upper_bound=2.4)
        elif isinstance(query, str):
            ans = self._kd_tree.query(self.virtual_and_interfacial_atoms[query],
                                      k=5,
                                      distance_upper_bound=2.4)
        else:
            raise TypeError('Wrong input type for query')

        ans1 = []

        for item in zip(ans[0], ans[1]):
            if 1.47 < item[0] < 2.4 and self.virtual_and_interfacial_atoms.keys()[item[1]].startswith("*"):
                ans1.append(item[1])

        return ans1


if __name__ == '__main__':

    sd = StructDesigner()
    print "Done!"

