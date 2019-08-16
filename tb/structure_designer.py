"""
The module contains classes serving to define
geometrical structure and boundary conditions of the problem.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import copy
from collections import OrderedDict
import logging
import numpy as np
import scipy.spatial
from scipy.spatial.distance import euclidean
from tb.aux_functions import xyz2np, count_species
from tb.abstract_interfaces import AbstractStructureDesigner
from tb.aux_functions import print_dict


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_spliting_indices(tree, level=3):
    length = len(tree.indices)
    combs = []

    for j1 in range(level):
        for comb in combs:
            trees = []
            for tree in comb:
                trees.append(item.lesser)
                trees.append(item.greater)
                trees.append(item.greater)
                trees.append(item.lesser)
            tree = trees

    combs = []

    for item in tree:
        combs.append(item.indices)

    combs = np.hstack(tuple(combs))

    ar = np.arange(0, len(combs))

    ans = []

    from sympy.utilities.iterables import multiset_permutations
    for p in multiset_permutations(ar):
        mylist = [combs[i] for i in p]
        ans.append(np.hstack(tuple(mylist)))

    return ans


def shift(mat):

    ans = np.zeros(mat.shape, dtype=np.int)

    cut = mat.shape[0] // 2

    ans[:cut] = mat[cut:]
    ans[cut:] = mat[:cut]

    return ans


def bandwidth1(mat):

    j = 0

    while np.count_nonzero((np.diag(mat, mat.shape[0] - j - 1))) == 0 and j < mat.shape[0]:
        j += 1

    return mat.shape[0] - j - 1


def bandwidth(mat):

    ans = 0

    for j in range(1, mat.shape[0]):
        if np.count_nonzero((np.diag(mat, j))) > 0:
            ans = j

    return ans


# Helper function to store the inroder traversal of a tree
def storeInorder(root, inorder):
    # Base Case
    if root is None:
        return

        # First store the left subtree
    storeInorder(root.left, inorder)

    # Copy the root's data
    inorder.append(root.data)

    # Finally store the right subtree
    storeInorder(root.right, inorder)


# A helper funtion to count nodes in a binary tree
def countNodes(root):
    if root is None:
        return 0

    return countNodes(root.lesser) + countNodes(root.greater) + 1


def is_in_coords(coord, coords):

    ans = False

    for xyz in list(coords):
        ans += (np.linalg.norm(coord - xyz) < 0.01)

    return ans


class StructDesignerXYZ(AbstractStructureDesigner):
    """
    The class builds an atomic structure from the xyz file or string.
    """

    def __init__(self, **kwargs):

        xyz = kwargs.get('xyz', '/home/mk/TB_project/tb/my_si.xyz')
        nn_distance = kwargs.get('nn_distance', 2.39)
        vec = kwargs.get('vec', 0)
        lead_l = kwargs.get('lead_l', 0)
        lead_r = kwargs.get('lead_r', 0)

        try:
            with open(xyz, 'r') as read_file:
                reader = read_file.read()
        except IOError:
            reader = xyz

        labels, coords = xyz2np(reader)

        logging.info("The xyz-file:\n {}".format(reader))
        logging.info("---------------------------------\n")

        self._nn_distance = nn_distance                            # maximal distance to a neighbor
        self._num_of_species = count_species(labels)               # dictionary of elements and
                                                                   # their number per unit cell
        self._num_of_nodes = sum(self.num_of_species.values())

        # if isinstance(vec, list):
        #     coords1 = copy.deepcopy(coords)
        #     coords1 = 100*(np.matrix(vec) * np.matrix(coords1).T).T
        #     coords1 = np.vstack((coords.T, coords1.T)).T
        #
        #     _kd_tree = scipy.spatial.cKDTree(coords1,
        #                                           leafsize=1,
        #                                           balanced_tree=True)
        #     indices = _kd_tree.indices
        #
        # elif isinstance(lead_l, list) and isinstance(lead_r, list):
        #
        #     gamma = 0.3 * np.min(np.diff(coords[:, 2]))
        #
        #     pot = np.zeros(coords.shape[0])
        #     for j, coord in enumerate(coords):
        #         for lll in lead_l:
        #             pot[j] -= 1.0 / (euclidean(coord, np.array(lll))**2 + gamma**2)
        #         for rrr in lead_r:
        #             pot[j] += 1.0 / (euclidean(coord, np.array(rrr))**2 + gamma**2)
        #
        #     indices = argsort(list(pot.tolist()))
        # else:
        #     indices = np.arange(len(coords))
        #
        # coords = coords[indices, :]
        # labels = list(np.array(labels)[indices])
        self._atom_list = OrderedDict(list(zip(labels, coords)))
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list.values())), leafsize=1, balanced_tree=True)

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

        ans = self._get_neighbours(query)

        ans1 = [ans[1][0]]

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance:
                ans1.append(item[1])

        return ans1


class CyclicTopology(AbstractStructureDesigner):
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

        self._kd_tree = scipy.spatial.cKDTree(list(self.virtual_and_interfacial_atoms.values()),
                                              leafsize=100, balanced_tree=True)

        logging.info("Primitive_cell_vectors: \n {} \n".format(primitive_cell_vectors))
        logging.info("Virtual and interfacial atoms: \n "
                     "{} ".format(print_dict(self.virtual_and_interfacial_atoms)))
        logging.info("---------------------------------\n")

    @property
    def atom_list(self):
        return self.virtual_and_interfacial_atoms

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

        distances1 = np.ones(distances1.shape)
        distances2 = np.ones(distances1.shape)

        # form new lists of atoms
        count = 0
        for j, item in enumerate(coords):

            if any(distances1[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)

                for surf in np.where(distances1[j])[0]:

                    count = self._translate_atom_1st_order(item,
                                                           np.array(self.pcv[surf]),
                                                           "_" + str(j) + "_" + labels[j],
                                                           coords,
                                                           count)

                    count = self._translate_atom_2d_order(item,
                                                          np.array(self.pcv[surf]),
                                                          "_" + str(j) + "_" + labels[j],
                                                          coords,
                                                          count)

            if any(distances2[j]):
                self.virtual_and_interfacial_atoms.update({str(j) + "_" + labels[j]: item})
                self.interfacial_atoms_ind.append(j)
                for surf in np.where(distances2[j])[0]:

                    count = self._translate_atom_1st_order(item,
                                                           -np.array(self.pcv[surf]),
                                                           "_" + str(j) + "_" + labels[j],
                                                           coords,
                                                           count)

                    count = self._translate_atom_2d_order(item,
                                                          -np.array(self.pcv[surf]),
                                                          "_" + str(j) + "_" + labels[j],
                                                          coords,
                                                          count)

        # remove non-unique elements
        self.interfacial_atoms_ind = list(set(self.interfacial_atoms_ind))

    def _translate_atom_1st_order(self, atom_coords, cell_vector, label, penalty_coords, count):

        try_coords = atom_coords + cell_vector

        if not is_in_coords(try_coords, penalty_coords) and \
                not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):
            self.virtual_and_interfacial_atoms.update({"*_" + str(count) + label: try_coords})
            count += 1

        return count

    def _translate_atom_2d_order(self, atom_coords, cell_vector, label, penalty_coords, count):

        for vec in self.pcv:

            try_coords = atom_coords + cell_vector + vec

            if not is_in_coords(try_coords, penalty_coords) and \
                    not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):

                self.virtual_and_interfacial_atoms.update({"**_" + str(count) + label: try_coords})
                count += 1

            try_coords = atom_coords + cell_vector - vec

            if not is_in_coords(try_coords, penalty_coords) and \
                    not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):

                self.virtual_and_interfacial_atoms.update({"**_" + str(count) + label: try_coords})
                count += 1

        return count

    def get_neighbours(self, query):

        ans = self._get_neighbours(query)

        ans1 = []

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance and \
                    list(self.virtual_and_interfacial_atoms.keys())[item[1]].startswith("*"):
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

    sd = StructDesignerXYZ(xyz='/home/mk/TB_project/input_samples/SiNW2.xyz')
    print("Done!")
