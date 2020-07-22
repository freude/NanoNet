"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
from __future__ import print_function
from __future__ import absolute_import
from itertools import product
import numpy as np
import yaml


def accum(accmap, input, func=None, size=None, fill_value=0, dtype=None):
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    input : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed. (Default value = None)
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`. (Default value = None)
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used. (Default value = None)

    Returns
    -------
    out : ndarray
        The accumulated results.
        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.

    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1, 2, 3], [4, -1, 6], [-1, 8, 9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    >>> s = accum(accmap,a)
    >>> s
    array([ 9,  7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([[[0,0],[0,0],[0,1]],[[0,0],[0,0],[0,1]],[[1,0],[1,0],[1,1]],])
    >>> # Accumulate using a product.
    >>> accum(accmap,a,func=prod,dtype=float)
    array([[-8., 18.],
           [-8.,  9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap,a,func=lambda x: x,dtype='O')
    array([[list([1, 2, 4, -1]), list([3, 6])],
           [list([-1, 8]), list([9])]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:input.ndim] != input.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = input.dtype
    if accmap.shape == input.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(input.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in input.shape]):
        indx = tuple(accmap[s])
        val = input[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out


def xyz2np(xyz):
    """Transforms xyz-file formatted string to lists of atomic labels and coordinates

    Parameters
    ----------
    xyz :
        xyz-formatted string

    Returns
    -------
    list, list
        list of labels and list of coordinates

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
    """From the list of labels creates a dictionary where the keys represents labels and
    values are numbers of their repetitions in the list

    Parameters
    ----------
    list_of_labels :
        return:

    Returns
    -------

    """
    counter = {}

    for item in list_of_labels:

        key = ''.join([i for i in item if not i.isdigit()])

        try:
            counter[key] += 1
        except KeyError:
            counter[key] = 1

    return counter


def get_k_coords(special_points, num_of_points, label):
    """Generates a array of the coordinates in the k-space from the set of
    high-symmetry points and number of nodes between them

    Parameters
    ----------
    special_points :
        list of labels for high-symmetry points
    num_of_points :
        list of node numbers in each section of the path in the k-space
    label :
        chemical element

    Returns
    -------
    numpy.ndarray
        array of coordinates in k-space

    """

    from nanonet.tb.special_points import SPECIAL_K_POINTS_BI, SPECIAL_K_POINTS_SI

    if isinstance(label, str):
        if label == 'Bi':
            SPECIAL_K_POINTS = SPECIAL_K_POINTS_BI
        if label == 'Si':
            SPECIAL_K_POINTS = SPECIAL_K_POINTS_SI
    else:
        SPECIAL_K_POINTS = label

    k_vectors = np.zeros((sum(num_of_points), 3))
    offset = 0

    for j in range(len(num_of_points)):
        sequence1 = np.linspace(SPECIAL_K_POINTS[special_points[j]][0],
                                SPECIAL_K_POINTS[special_points[j + 1]][0], num_of_points[j])
        sequence2 = np.linspace(SPECIAL_K_POINTS[special_points[j]][1],
                                SPECIAL_K_POINTS[special_points[j + 1]][1], num_of_points[j])
        sequence3 = np.linspace(SPECIAL_K_POINTS[special_points[j]][2],
                                SPECIAL_K_POINTS[special_points[j + 1]][2], num_of_points[j])

        k_vectors[offset:offset + num_of_points[j], :] = \
            np.vstack((sequence1, sequence2, sequence3)).T

        offset += num_of_points[j]

    return k_vectors


def dict2xyz(input_data):
    """

    Parameters
    ----------
    input_data :
        return:

    Returns
    -------

    """

    if not isinstance(input_data, dict):
        return input_data

    output = str(input_data['num_atoms']) + '\n'
    output += str(input_data['title']) + '\n'

    for j in range(input_data['num_atoms']):
        output += list(input_data['atoms'][j].keys())[0] + \
                  "    " + str(list(input_data['atoms'][j].values())[0][0]) + \
                  "    " + str(list(input_data['atoms'][j].values())[0][1]) + \
                  "    " + str(list(input_data['atoms'][j].values())[0][2]) + "\n"

    return output


def yaml_parser(input_data):
    """

    Parameters
    ----------
    input_data :
        return:

    Returns
    -------

    """

    output = None

    if input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.safe_load(input_data)
        except yaml.YAMLError as exc:
            print(exc)

    output['primitive_cell'] = np.array(output['primitive_cell']) * output['lattice_constant']

    return output


def print_table(myDict, colList=None, sep='\uFFFA'):
    """Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    sep: row separator. Ex: sep='\n' on Linux. Default: dummy to not split line.
    Author: Thierry Husson - Use it as you want but don't blame me.

    Parameters
    ----------
    myDict :
        
    colList :
         (Default value = None)
    sep :
         (Default value = '\uFFFA')

    Returns
    -------

    """

    if not colList:
        colList = list(myDict[0].keys() if myDict else [])

    myList = [colList]  # 1st row = header

    for item in myDict:
        myList.append([str(item[col]) for col in colList])

    colSize = [max(map(len, (sep.join(col)).split(sep))) for col in zip(*myList)]

    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    line = formatStr.replace(' | ', '-+-').format(*['-' * i for i in colSize])
    item = myList.pop(0)
    lineDone = False

    out = "\n"

    while myList:
        if all(not i for i in item):
            item = myList.pop(0)
            if line and (sep != '\uFFFA' or not lineDone):
                out += line
                out += "\n"
                lineDone = True

        row = [i.split(sep, 1) for i in item]
        out += formatStr.format(*[i[0] for i in row])
        out += "\n"
        item = [i[1] if len(i) > 1 else '' for i in row]

    out += line
    out += "\n"

    return out


def print_dict(dictionary):
    """

    Parameters
    ----------
    dictionary :
        

    Returns
    -------

    """
    out = "{:<18} {:<15} \n".format('Label', 'Coordinates')
    for key, value in dictionary.items():
        out += "{:<18} {:<15} \n".format(key, str(value))

    return out


# def split_into_subblocks(h_0, h_l, h_r):
#     """
#     Split Hamiltonian matrix and coupling matrices into subblocks
#
#     :param h_0:                     Hamiltonian matrix
#     :param h_l:                     left inter-cell coupling matrices
#     :param h_r:                     right inter-cell coupling matrices
#     :return h_0_s, h_l_s, h_r_s:    lists of subblocks
#     """
#
#     def find_nonzero_lines(mat, order):
#
#         if order == 'top':
#             line = mat.shape[0]
#             while line > 0:
#                 if np.count_nonzero(mat[line - 1, :]) == 0:
#                     line -= 1
#                 else:
#                     break
#         elif order == 'bottom':
#             line = -1
#             while line < mat.shape[0] - 1:
#                 if np.count_nonzero(mat[line + 1, :]) == 0:
#                     line += 1
#                 else:
#                     line = mat.shape[0] - (line + 1)
#                     break
#         elif order == 'left':
#             line = mat.shape[1]
#             while line > 0:
#                 if np.count_nonzero(mat[:, line - 1]) == 0:
#                     line -= 1
#                 else:
#                     break
#         elif order == 'right':
#             line = -1
#             while line < mat.shape[1] - 1:
#                 if np.count_nonzero(mat[:, line + 1]) == 0:
#                     line += 1
#                 else:
#                     line = mat.shape[1] - (line + 1)
#                     break
#         else:
#             raise ValueError('Wrong value of the parameter order')
#
#         return line
#
#     h_0_s = []
#     h_l_s = []
#     h_r_s = []
#
#     if isinstance(h_l, np.ndarray) and isinstance(h_r, np.ndarray):
#         h_r_h = find_nonzero_lines(h_r, 'bottom')
#         h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
#         h_l_h = find_nonzero_lines(h_l, 'top')
#         h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')
#
#     if isinstance(h_l, int) and isinstance(h_r, int):
#         h_l_h = h_l
#         h_r_v = h_l
#         h_r_h = h_r
#         h_l_v = h_r
#
#     edge, edge1 = compute_edge(h_0)
#
#     left_block = max(h_l_h, h_r_v)
#     right_block = max(h_r_h, h_l_v)
#
#     from nanonet.tb.sorting_algorithms import split_matrix_0, cut_in_blocks, split_matrix
#
#     blocks = split_matrix_0(h_0, left=left_block, right=right_block)
#     h_0_s, h_l_s, h_r_s = cut_in_blocks(h_0, blocks)
#
#     # blocks = compute_blocks(left_block, right_block, edge, edge1)
#     # j1 = 0
#     #
#     # for j, block in enumerate(blocks):
#     #     h_0_s.append(h_0[j1:block + j1, j1:block + j1])
#     #     if j < len(blocks) - 1:
#     #         h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
#     #         h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
#     #     j1 += block
#
#     return h_0_s, h_l_s, h_r_s, blocks


def compute_edge(mat):
    """

    Parameters
    ----------
    mat :
        

    Returns
    -------

    """
    # First get some statistics
    row, col = np.where(mat != 0.0)  # Output rows and columns of all non-zero elements.

    # Clever use of accumarray:
    outeredge = accum(row, col, np.max) + 1
    outeredge[0] = max(0, outeredge[0])
    outeredge = np.maximum.accumulate(outeredge)

    outeredge1 = accum(np.max(row) - row[::-1], np.max(row) - col[::-1], np.max) + 1
    outeredge1[0] = max(0, outeredge1[0])
    outeredge1 = np.maximum.accumulate(outeredge1)

    return outeredge, outeredge1


def blocksandborders_constrained(left_block, right_block, edge, edge1):
    """A version of blocksandborders with constraints - periodic boundary conditions.

    Parameters
    ----------
    mat :
        input matrix
    left_block :
        left block constrained minimal size
    right_block :
        right block constrained minimal size
    edge :
        edge of sparsity pattern
    edge1 :
        conjugate edge of sparsity pattern

    Returns
    -------
    type
        array of diagonal block sizes

    """

    size = len(edge)
    left_block = max(1, left_block)
    right_block = max(1, right_block)

    if left_block + right_block < size:                               # if blocks do not overlap

        new_left_block = edge[left_block - 1] - left_block
        new_right_block = edge1[right_block - 1] - right_block
        #
        # new_right_block = np.max(np.argwhere(np.abs(edge - (size - right_block)) -
        #                                      np.min(np.abs(edge - (size - right_block))) == 0)) + 1
        # new_right_block = size - new_right_block - right_block

        if left_block + new_left_block <= size - right_block and\
                size - right_block - new_right_block >= left_block:    # spacing between blocks is sufficient

            blocks = blocksandborders_constrained(new_left_block,
                                                  new_right_block,
                                                  edge[left_block:-right_block] - left_block,
                                                  edge1[right_block:-left_block] - right_block)

            return [left_block] + blocks + [right_block]
        else:
            if new_left_block > new_right_block:
                return [left_block] + [size - left_block]
            else:
                return [size - right_block] + [right_block]

    elif left_block + right_block == size:                            # sum of blocks equal to the matrix size
        return [left_block] + [right_block]
    else:                                                             # blocks overlap
        return [size]


def argsort(seq):
    """

    Parameters
    ----------
    seq :
        

    Returns
    -------

    """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def shift(mat):
    """

    Parameters
    ----------
    mat :
        

    Returns
    -------

    """

    ans = np.zeros(mat.shape, dtype=np.int)

    cut = mat.shape[0] // 2

    ans[:cut] = mat[cut:]
    ans[cut:] = mat[:cut]

    return ans


def bandwidth1(mat):
    """

    Parameters
    ----------
    mat :
        

    Returns
    -------

    """

    j = 0

    while np.count_nonzero((np.diag(mat, mat.shape[0] - j - 1))) == 0 and j < mat.shape[0]:
        j += 1

    return mat.shape[0] - j - 1


def bandwidth(mat):
    """

    Parameters
    ----------
    mat :
        

    Returns
    -------

    """

    ans = 0

    for j in range(1, mat.shape[0]):
        if np.count_nonzero((np.diag(mat, j))) > 0:
            ans = j

    return ans


# Helper function to store the inroder traversal of a tree
def storeInorder(root, inorder):
    """

    Parameters
    ----------
    root :
        
    inorder :
        

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    root :
        

    Returns
    -------

    """
    if root is None:
        return 0

    return countNodes(root.lesser) + countNodes(root.greater) + 1


def is_in_coords(coord, coords):
    """

    Parameters
    ----------
    coord :
        
    coords :
        

    Returns
    -------

    """

    ans = False

    for xyz in list(coords):
        ans += (np.linalg.norm(coord - xyz) < 0.01)

    return ans