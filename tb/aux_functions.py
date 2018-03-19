"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
from itertools import islice
import numpy as np
from params import SPECIAL_K_POINTS


def xyz2np(xyz):
    """
    Transforms xyz-file formatted string to lists of atomic labels and coordinates

    :param xyz:  xyz-formatted string
    :return:     list of labels and list of coordinates
    :rtype:      list, list
    """

    xyz = xyz.splitlines()
    num_of_atoms = int(xyz[0])
    ans = np.zeros((num_of_atoms, 3))
    j = 0
    atoms = []

    for line in islice(xyz, 2, num_of_atoms + 2):
        temp = line.split()
        atoms.append(temp[0])
        ans[j, 0] = float(temp[1])
        ans[j, 1] = float(temp[2])
        ans[j, 2] = float(temp[3])
        j += 1

    return atoms, ans


def count_species(list_of_labels):
    """
    From the list of labels creates a dictionary where the keys represents labels and
    values are numbers of their repetitions in the list

    :param list_of_labels:
    :return:
    """
    counter = {}

    for item in list_of_labels:
        try:
            counter[item[0]] += 1
        except KeyError:
            counter[item[0]] = 1

    return counter


def get_k_coords(special_points, num_of_points):
    """
    Generates a array of the coordinates in the k-space from the set of
    high-symmetry points and number of nodes between them

    :param special_points:   list of labels for high-symmetry points
    :param num_of_points:    list of node numbers in each section of the path in the k-space
    :return:                 array of coordinates in k-space
    :rtype:                  numpy.ndarray
    """

    k_vectors = np.zeros((sum(num_of_points), 3))
    offset = 0

    for j in xrange(len(num_of_points)):

        sequence1 = np.linspace(SPECIAL_K_POINTS[special_points[j]][0],
                                SPECIAL_K_POINTS[special_points[j+1]][0], num_of_points[j])
        sequence2 = np.linspace(SPECIAL_K_POINTS[special_points[j]][1],
                                SPECIAL_K_POINTS[special_points[j+1]][1], num_of_points[j])
        sequence3 = np.linspace(SPECIAL_K_POINTS[special_points[j]][2],
                                SPECIAL_K_POINTS[special_points[j+1]][2], num_of_points[j])

        k_vectors[offset:offset + num_of_points[j], :] = \
            np.vstack((sequence1, sequence2, sequence3)).T

        offset += num_of_points[j]

    return k_vectors


if __name__ == "__main__":

    sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
    num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]

    k_points = get_k_coords(sym_points, num_points)
    print k_points
