"""
The module contains classes serving to define
geometrical structure and boundary conditions of the problem.
"""
from collections import OrderedDict
import numpy as np
import scipy.spatial
# from third_party.cluster_100 import supercell
from aux_functions import xyz2np, count_species
from abstract_interfaces import AbstractStructureDesigner


class StructDesignerXYZ(AbstractStructureDesigner):
    """
    The class builds the atomic structure from the xyz file or string.
    """

    def __init__(self, xyz='/home/mk/TB_project/tb/my_si.xyz', nn_distance=2.39):

        try:
            with open(xyz, 'rb') as read_file:
                reader = read_file.read()

        except IOError:
            reader = xyz

        labels, coords = xyz2np(reader)
        self._nn_distance = nn_distance
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
                                      distance_upper_bound=self._nn_distance)
        elif isinstance(query, int):
            ans = self._kd_tree.query(self.atom_list.items()[query][1],
                                      k=5,
                                      distance_upper_bound=self._nn_distance)
        elif isinstance(query, str):
            ans = self._kd_tree.query(self.atom_list[query],
                                      k=5,
                                      distance_upper_bound=self._nn_distance)
        else:
            raise TypeError('Wrong input type for query')

        ans1 = [ans[1][0]]

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.25 < item[0] < self._nn_distance:
                ans1.append(item[1])

        return ans1


# class StructDesigner(StructDesignerXYZ):
#     """
#     The class builds the atomic structure using
#     a third party generating function supercell
#     from the module cluster100.
#     """
#
#     def __init__(self):
#
#         a_si = 5.50
#
#         si = supercell(2, 2, 2, AFinite=True, BFinite=True, CFinite=False, RemoveSilane=False)
#         si.scale(a_si)
#         # si.translate(-A*aSi/2,-B*aSi/2,-C*aSi/2)
#         # si.translate(0,0,-C*aSi/2)
#         # si.translate(0, 0, 0)
#
#         if si.closeatoms():
#             print "Close contacts: ", si.closeatoms()
#         a = si.toxyz()
#
#         open('si.xyz', 'w').write(a)
#
#         labels, coords = xyz2np(a)
#         self._num_of_species = count_species(labels)
#         self._num_of_nodes = sum(self.num_of_species.values())
#         self._atom_list = OrderedDict(zip(labels, coords))
#         self._kd_tree = scipy.spatial.cKDTree(coords, leafsize=100)


class CyclicTopology(object):
    """
    The class provides functionality for determining
    the periodic boundary conditions for a crystal cell.
    The object of the class is instantiated by
    a set of the primitive cell vectors.
    """

    def __init__(self, primitive_cell_vectors, labels, coords, nn_distance):

        self._nn_distance = nn_distance
        self.pcv = primitive_cell_vectors

        # compute vectors' lengths
        self.sizes = []
        for item in self.pcv:
            self.sizes.append(np.linalg.norm(item))

        self.interfacial_atoms_ind = []
        self.virtual_and_interfacial_atoms = OrderedDict()
        self._generate_atom_list(labels, coords)

        self._kd_tree = scipy.spatial.cKDTree(self.virtual_and_interfacial_atoms.values(),
                                              leafsize=100)

    def _generate_atom_list(self, labels, coords):
        """

        :param labels:    labels of atoms
        :param coords:    coordinates of atoms
        :return:
        """

        # matrices of distances between atoms and interfaces
        distances1 = np.empty((len(coords), len(self.pcv)), dtype=np.float)
        distances2 = np.empty((len(coords), len(self.pcv)), dtype=np.float)

        for j1, coord in enumerate(coords):    # for each atom in the unit cell
            for j2, basis_vec in enumerate(self.pcv):    # for lattice basis vector

                # compute distance to the primary plane of the unit cell
                distances1[j1, j2] = np.inner(coord, basis_vec) / self.sizes[j2]
                # compute distance to the adjacent plane of the  unit cell
                distances2[j1, j2] = np.inner(coord - basis_vec, basis_vec) / self.sizes[j2]

        # transform distance to the boolean variable defining whether atom belongs to the interface or not
        distances1 = np.abs(distances1 - np.min(distances1)) < self._nn_distance * 0.25
        distances2 = np.abs(np.abs(distances2) - np.min(np.abs(distances2))) < self._nn_distance * 0.25

        # form new lists of atoms
        count = 0
        for j, item in enumerate(coords):

            if any(distances1[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)

                for surf in np.where(distances1[j])[0]:
                    atom_coords = item + self.pcv[surf]
                    self.virtual_and_interfacial_atoms.update({"*_" + str(count) +
                                                               "_" + str(j) + "_" +
                                                               labels[j]: atom_coords})
                    count += 1

            if any(distances2[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)
                for surf in np.where(distances2[j])[0]:
                    atom_coords = item - self.pcv[surf]
                    self.virtual_and_interfacial_atoms.update({"*_" + str(count) +
                                                               "_" + str(j) + "_" +
                                                               labels[j]: atom_coords})
                    count += 1

        # remove non-unique elements
        self.interfacial_atoms_ind = list(set(self.interfacial_atoms_ind))

    def get_neighbours(self, query):

        if isinstance(query, list) or isinstance(query, np.ndarray):
            ans = self._kd_tree.query(query,
                                      k=5,
                                      distance_upper_bound=self._nn_distance)
        elif isinstance(query, int):
            ans = self._kd_tree.query(self.virtual_and_interfacial_atoms.items()[query][1],
                                      k=5,
                                      distance_upper_bound=self._nn_distance)
        elif isinstance(query, str):
            ans = self._kd_tree.query(self.virtual_and_interfacial_atoms[query],
                                      k=5,
                                      distance_upper_bound=self._nn_distance)
        else:
            raise TypeError('Wrong input type for query')

        ans1 = []

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.25 < item[0] < self._nn_distance and \
                    self.virtual_and_interfacial_atoms.keys()[item[1]].startswith("*"):
                ans1.append(item[1])

        return ans1

    @staticmethod
    def atom_classifier(coords, leads):

        distance_to_surface1 = np.inner(coords, leads) / np.linalg.norm(leads)
        distance_to_surface2 = np.inner(coords - leads, leads) / np.linalg.norm(leads)

        flag = None

        if distance_to_surface1 < 0:
            flag = 'L'
        if distance_to_surface2 >= 0:
            flag = 'R'

        return flag


if __name__ == '__main__':

    sd = StructDesignerXYZ()
    print "Done!"

