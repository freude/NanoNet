"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
import numpy as np
import yaml
from constants import SPECIAL_K_POINTS


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
    unique_labels = dict()

    for line in xyz[2:]:
        if len(line.strip()) > 0:
            temp = line.split()
            label = ''.join([i for i in temp[0] if not i.isdigit()])

            try:
                unique_labels[label] += 1
                temp[0] = label + str(unique_labels[label])
            except KeyError:
                temp[0] = label + '1'
                unique_labels[label] = 1

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

        key = ''.join([i for i in item if not i.isdigit()])

        try:
            counter[key] += 1
        except KeyError:
            counter[key] = 1

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


def dict2xyz(input_data):
    """

    :param input_data:
    :return:
    """

    if not isinstance(input_data, dict):
        return input_data

    output = str(input_data['num_atoms']) + '\n'
    output += str(input_data['title']) + '\n'

    for j in xrange(input_data['num_atoms']):
        output += input_data['atoms'][j].keys()[0] + \
                  "    " + str(input_data['atoms'][j].values()[0][0]) +\
                  "    " + str(input_data['atoms'][j].values()[0][1]) + \
                  "    " + str(input_data['atoms'][j].values()[0][2]) + "\n"

    return output


def yaml_parser(input_data):
    """

    :param input_data:
    :return:
    """

    output = None

    if input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.load(input_data)
        except yaml.YAMLError as exc:
            print(exc)

    output['primitive_cell'] = np.array(output['primitive_cell']) * output['lattice_constant']

    return output


if __name__ == "__main__":

    sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
    num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]

    k_points = get_k_coords(sym_points, num_points)
    print k_points
