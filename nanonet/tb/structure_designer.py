"""
The module contains classes defining
geometrical structure and boundary conditions for the tight-binding model.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from collections import OrderedDict
import logging
import numpy as np
import scipy.spatial
from nanonet.tb.abstract_interfaces import AbstractStructureDesigner
from nanonet.tb.aux_functions import xyz2np, count_species, is_in_coords, print_dict


class StructDesignerXYZ(AbstractStructureDesigner):
    """The class builds an atomic structure from either
    the filename of a xyz-file or
    xyz data itself represented as a Python string.
    The class arrange atomic coordinates in kd-tree and
    sorts them if needed according to a specified sorting procedure.

    Parameters
    ----------

    Returns
    -------

    Attributes
    ----------
    nn_distance : float
        nearest neighbour search radius (default 0)
    num_of_species : int
        number of chemical elements corresponding to the number of distinct basis sets.
    atom_list : OrderedDict
        list of atomic species and their coordinates
    kd_tree : scipy.spatial.ckdtree.cKDTree
        kd-tree for fast nearest-neighbour search
    left_lead : list
        list of atomic indices connected to the left lead,
        needed for sorting atomic coordinates (default [])
    right_lead : list
        list of atomic indices connected to the right lead,
        needed for sorting atomic coordinates (default [])
    sort_func : func
        function for sorting atomic coordinates (default None)
    """

    def __init__(self, **kwargs):
        # ------------ parse xyz file or string --------------
        xyz = kwargs.get('xyz', None)

        try:
            with open(xyz, 'r') as read_file:
                reader = read_file.read()
        except IOError:
            reader = xyz

        labels, coords = xyz2np(reader)

        num_lines = reader.count('\n')
        if num_lines > 11:
            logging.info("The xyz-file:\n {}".format('\n'.join(reader.split('\n')[:11])))
            logging.info("                  .                    ")
            logging.info("                  .                    ")
            logging.info("                  .                    ")
            logging.info("There are {} more coordinates".format(str(num_lines-10)))
        else:
            logging.info("The xyz-file:\n {}".format(reader))
        logging.info("---------------------------------\n")

        # ------------- count species and nodes --------------
        self._nn_distance = kwargs.get('nn_distance', 0)           # maximal distance to a neighbor
        self._num_of_species = count_species(labels)               # dictionary of elements and
                                                                   # their number per unit cell
        self._num_of_nodes = sum(self.num_of_species.values())
        # ------- make list of coordinates and kd-tree -------
        self._atom_list = OrderedDict(list(zip(labels, coords)))
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list.values())),
                                              leafsize=1,
                                              balanced_tree=True)

        # ------- rearrange list of atomic coordinates -------
        self.left_lead = kwargs.get('left_lead', [])
        self.right_lead = kwargs.get('right_lead', [])
        self.sort_func = kwargs.get('sort_func', None)
        self.reorder = None

        if self.sort_func is not None:
            self._sort(labels, coords)

    def _sort(self, labels, coords):
        """

        Parameters
        ----------
        labels :
            
        coords :
            

        Returns
        -------

        """

        coords = np.array(coords)
        h_matrix = np.zeros((coords.shape[0], coords.shape[0]))

        self._nn_distance = 2 * self._nn_distance
        for j in range(len(coords)):
            ans = self.get_neighbours(j)
            h_matrix[j, ans] = 1

        self._nn_distance = self._nn_distance / 2

        indices = self.sort_func(coords=coords,
                                 left_lead=self.left_lead,
                                 right_lead=self.right_lead,
                                 mat=h_matrix)

        self.reorder = indices
        if (isinstance(self.left_lead, list) or isinstance(self.left_lead, np.ndarray)) and len(self.left_lead) > 0:
            self.left_lead = np.squeeze(np.concatenate([np.where(indices == item) for item in self.left_lead]))
        if (isinstance(self.right_lead, list) or isinstance(self.right_lead, np.ndarray)) and len(self.right_lead) > 0:
            self.right_lead = np.squeeze(np.concatenate([np.where(indices == item) for item in self.right_lead]))
        coords = coords[indices]
        labels = [labels[i] for i in indices]

        self._atom_list = OrderedDict(list(zip(labels, coords)))
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list.values())), leafsize=1, balanced_tree=True)

    def add_leads(self, left_lead, right_lead):
        """

        Parameters
        ----------
        left_lead :
            
        right_lead :
            

        Returns
        -------

        """
        self.left_lead = left_lead
        self.right_lead = right_lead

    @property
    def atom_list(self):
        """ """
        return self._atom_list

    @property
    def num_of_nodes(self):
        """ """
        return self._num_of_nodes

    @property
    def num_of_species(self):
        """ """
        return self._num_of_species

    def get_neighbours(self, query):
        """

        Parameters
        ----------
        query :
            

        Returns
        -------

        """

        ans = self._get_neighbours(query)

        ans1 = [ans[1][0]]

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance:
                ans1.append(item[1])

        return ans1


class CyclicTopology(AbstractStructureDesigner):
    """The class provides functionality for determining
    the periodic boundary conditions for a crystal cell.
    The object of the class is instantiated by
    a set of the primitive cell vectors.

    Parameters
    ----------

    Returns
    -------

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
        logging.debug("Virtual and interfacial atoms: \n "
                      "{} ".format(print_dict(self.virtual_and_interfacial_atoms)))
        logging.info("---------------------------------\n")

    @property
    def atom_list(self):
        """ """
        return self.virtual_and_interfacial_atoms

    def _generate_atom_list(self, labels, coords):
        """

        Parameters
        ----------
        labels :
            labels of atoms
        coords :
            coordinates of atoms

        Returns
        -------

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

        # distances1 = np.ones(distances1.shape)
        # distances2 = np.ones(distances1.shape)

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
                                                           -1 * np.array(self.pcv[surf]),
                                                           "_" + str(j) + "_" + labels[j],
                                                           coords,
                                                           count)

                    count = self._translate_atom_2d_order(item,
                                                          -1 * np.array(self.pcv[surf]),
                                                          "_" + str(j) + "_" + labels[j],
                                                          coords,
                                                          count)

        # remove non-unique elements
        self.interfacial_atoms_ind = list(set(self.interfacial_atoms_ind))

    def _translate_atom_1st_order(self, atom_coords, cell_vector, label, penalty_coords, count):
        """

        Parameters
        ----------
        atom_coords :
            
        cell_vector :
            
        label :
            
        penalty_coords :
            
        count :
            

        Returns
        -------

        """

        try_coords = atom_coords + cell_vector

        if not is_in_coords(try_coords, penalty_coords) and \
                not is_in_coords(try_coords, np.array(list(self.virtual_and_interfacial_atoms.values()))):
            self.virtual_and_interfacial_atoms.update({"*_" + str(count) + label: try_coords})
            count += 1

        return count

    def _translate_atom_2d_order(self, atom_coords, cell_vector, label, penalty_coords, count):
        """

        Parameters
        ----------
        atom_coords :
            
        cell_vector :
            
        label :
            
        penalty_coords :
            
        count :
            

        Returns
        -------

        """

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
        """

        Parameters
        ----------
        query :
            

        Returns
        -------

        """

        ans = self._get_neighbours(query)

        ans1 = []

        for item in zip(ans[0], ans[1]):
            if self._nn_distance * 0.1 < item[0] < self._nn_distance and \
                    list(self.virtual_and_interfacial_atoms.keys())[item[1]].startswith("*"):
                ans1.append(item[1])

        return ans1

    @staticmethod
    def atom_classifier(coords, leads):
        """

        Parameters
        ----------
        coords :
            
        leads :
            

        Returns
        -------

        """

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
    sd._sort([np.argmin(np.array(list(sd.atom_list.values()))[:, 2])],
             [np.argmax(np.array(list(sd.atom_list.values()))[:, 2])])
    print("Done!")
