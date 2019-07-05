"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
from __future__ import print_function
from __future__ import absolute_import
from itertools import product
import numpy as np
import yaml


def accum(accmap, input, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

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
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

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


def get_k_coords(special_points, num_of_points, label):
    """
    Generates a array of the coordinates in the k-space from the set of
    high-symmetry points and number of nodes between them

    :param special_points:   list of labels for high-symmetry points
    :param num_of_points:    list of node numbers in each section of the path in the k-space
    :param label:            chemical element
    :return:                 array of coordinates in k-space
    :rtype:                  numpy.ndarray
    """

    from tb.special_points import SPECIAL_K_POINTS_BI, SPECIAL_K_POINTS_SI

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

    :param input_data:
    :return:
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

    :param input_data:
    :return:
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
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    sep: row separator. Ex: sep='\n' on Linux. Default: dummy to not split line.
    Author: Thierry Husson - Use it as you want but don't blame me.
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
    out = "{:<18} {:<15} \n".format('Label', 'Coordinates')
    for key, value in dictionary.items():
        out += "{:<18} {:<15} \n".format(key, str(value))

    return out


def bandwidth(mat):
    j = 0

    while np.count_nonzero((np.diag(mat, mat.shape[0] - j - 1))) == 0 and j < mat.shape[0]:
        j += 1

    return mat.shape[0] - j - 1


# def blocksandborders(A):
#     """This is a function designed to output the blocks from a block-tridiagonal
#     matrix, A. This function assumes that bandwidth minimization has already
#     occurred and that this is to be applied to some matrix. There exist many
#     algorithms to reduce bandwidth, this is only to produce blocks. This
#     problem has likely been solved before, but the author could not find any
#     examples, likely due to insufficient knowledge of the literature.
#
#     Jesse Vaitkus, 2019
#
#     :param A:             input matrix
#     :return:              array of diagonal block sizes
#     """
#
#     # First get some statistics
#     sza = A.shape[0]  # Get the dimension of the input matrix
#     bandA = bandwidth(A)  # Bandwidth of matrix
#
#     row, col = np.where(A != 0.0)  # Output rows and columns of all non-zero elements.
#
#     # Clever use of accumarray:
#     outeredge = accum(row, col, np.max)
#     # accumarray takes _unordered_ data and bins it, by using a function like
#     # @max it then applies that function to the binned data. This tells us the
#     # largest index that is nonzero, AND even tells us which rows have zero
#     # non-zero elements.
#
#     # Fringe case, we make sure that the first element is always at least 1.
#     outeredge[0] = max(0, outeredge[0])
#
#     # Now we loop over all the outer indices, if an outer index is smaller than
#     # the previous, we increase it, this gives us an effective "outer edge" of
#     # our matrix, which is useful for my block detection method.
#
#     outeredge = np.maximum.accumulate(outeredge)  # following commented code is the same as cummax
#
#     # Now that we have our outer edges we can work out the blocks. We do not
#     # have a good guess as to what the first block should be a priori. What we
#     # do know is that no block will ever be larger than the bandwidth. This is
#     # our upper bound. In the ideal case the matrix is tridiagonal (not block
#     # tridiagonal) and so our lower bound is 1.
#
#     testindices = np.arange(0, bandA)
#
#     # Now we don't know how many blocks there will be, so we overallocate in
#     # preparation and trim afterwards. This doesn't cost anything meaningful,
#     # and saves time from the possible matrix size increase.
#     blocks = gen_blocks(outeredge, sza)
#     blocks = np.array(blocks).T
#     # The optimal block choice is the one that minimizes the sum of cubes. As
#     # this is the scaling of the recursive Green's function algorithm.
#     best = np.argmin(np.sum(blocks ** 3, 0))
#     blocks = blocks[:, best]
#     blocks = [item for item in blocks if item != 0]
#
#     # This last step is mainly a bookkeeping one, we add indices for the rows
#     # that the edges correspond to, this makes plotting them easier.
#     outeredge = [np.arange(0, sza), outeredge]
#
#     return blocks, outeredge
#
#
# def blocksandborders_constrained(A):
#     """A version of blocksandborders with constraints - periodic boundary conditions.
#
#     :param A:             input matrix
#     :return:              array of diagonal block sizes
#     """
#
#     outeredge = compute_edge(A)
#
#     blocks = []
#     unique_edges, index = np.unique(outeredge, return_inverse=True)
#     outeredges = []
#     outeredges.append(outeredge)
#
#     for j1 in range(1, 5):
#         for j in range(len(unique_edges[unique_edges < 3 * A.shape[0] / 4]) - j1):
#             temp_edge = np.copy(unique_edges)
#             temp_edge[j] = temp_edge[j + j1]
#             temp_edge = np.maximum.accumulate(temp_edge)
#             outeredges.append(temp_edge[index])
#
#     sza = A.shape[0]  # Get the dimension of the input matrix
#
#     for outeredge in outeredges:
#         blocks += gen_blocks(outeredge, sza)
#
#     blocks = np.array(blocks).T
#     indicies = []
#
#     for j in range(blocks.shape[1]):
#         cumsum = np.cumsum(blocks[:, j])
#         if A.shape[0] // 2 not in cumsum:
#             blocks[:, j] = 0
#             indicies.append(j)
#         else:
#             inds = np.where(cumsum == A.shape[0] // 2)[0]
#
#             for ind in inds:
#                 if blocks[0, j] != blocks[ind + 1, j]:
#                     blocks[:, j] = 0
#                     indicies.append(j)
#
#     blocks = np.delete(blocks, indicies, 1)
#     if blocks.tolist():
#         best = np.argmin(np.sum(blocks ** 3, 0))
#         blocks = blocks[:, best]
#         blocks = [item for item in blocks if item != 0]
#         outeredge = [np.arange(0, sza), outeredge]
#
#     return blocks, outeredge


def split_matrix(h_0):

    edge, edge1 = compute_edge(h_0)

    x = np.arange(1, np.max(edge - np.linspace(0, len(edge), len(edge), dtype=np.int)))
    y = np.arange(1, np.max(edge1 - np.linspace(0, len(edge), len(edge), dtype=np.int)))
    # X, Y = np.meshgrid(x, y)
    # init_blocks = np.vstack((X.flatten(), Y.flatten())).T
    blocks = []
    metric = []
    metrics = np.zeros((len(x), len(y)))

    for j1, item1 in enumerate(x):
        for j2, item2 in enumerate(y):
            block = blocksandborders_constrained(item1, item2, edge, edge1)
            blocks.append(block)
            metric.append(np.sum(np.array(block) ** 3))
            metrics[j1, j2] = np.sum(np.array(block) ** 3)

    j1 = 0
    h_0_s = []
    h_l_s = []
    h_r_s = []

    best = np.argmin(metric)
    blocks = blocks[best]
    # blocks = blocks[100]

    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block

    return h_0_s, h_l_s, h_r_s, blocks


def split_into_subblocks(h_0, h_l, h_r):
    """
    Split Hamiltonian matrix and coupling matrices into subblocks

    :param h_0:                     Hamiltonian matrix
    :param h_l:                     left inter-cell coupling matrices
    :param h_r:                     right inter-cell coupling matrices
    :return h_0_s, h_l_s, h_r_s:    lists of subblocks
    """

    def find_nonzero_lines(mat, order):

        if order == 'top':
            line = mat.shape[0]
            while line > 0:
                if np.count_nonzero(mat[line - 1, :]) == 0:
                    line -= 1
                else:
                    break
        elif order == 'bottom':
            line = -1
            while line < mat.shape[0] - 1:
                if np.count_nonzero(mat[line + 1, :]) == 0:
                    line += 1
                else:
                    line = mat.shape[0] - (line + 1)
                    break
        elif order == 'left':
            line = mat.shape[1]
            while line > 0:
                if np.count_nonzero(mat[:, line - 1]) == 0:
                    line -= 1
                else:
                    break
        elif order == 'right':
            line = -1
            while line < mat.shape[1] - 1:
                if np.count_nonzero(mat[:, line + 1]) == 0:
                    line += 1
                else:
                    line = mat.shape[1] - (line + 1)
                    break
        else:
            raise ValueError('Wrong value of the parameter order')

        return line

    h_0_s = []
    h_l_s = []
    h_r_s = []

    h_r_h = find_nonzero_lines(h_r, 'bottom')
    h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
    h_l_h = find_nonzero_lines(h_l, 'top')
    h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')

    edge, edge1 = compute_edge(h_0)

    left_block = max(h_l_h, h_r_v)
    right_block = max(h_r_h, h_l_v)

    blocks = blocksandborders_constrained(left_block, right_block, edge, edge1)
    j1 = 0

    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block

    return h_0_s, h_l_s, h_r_s, blocks


def compute_edge(mat):
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

    :param mat:                    input matrix
    :param left_block:             left block constrained minimal size
    :param right_block:            right block constrained minimal size
    :param edge:                   edge of sparsity pattern
    :param edge1:                  conjugate edge of sparsity pattern

    :return:                       array of diagonal block sizes
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


# def gen_blocks(outeredge, sza):
#     """
#     Computes decomposition of matrix into blocks
#
#     :param outeredge:       the upper edge of the sparsity pattern
#     :param sza:             maximal allowed size of the first block
#
#     :return:                array of diagonal block sizes
#     """
#
#     blocks = []
#     u_outeredge = np.where(np.abs(np.diff(outeredge)) > 1)[0] + 1
#
#     for edge_num in range(len(u_outeredge) // 2):
#
#         block = np.zeros(sza, dtype=np.int)
#
#         block[0] = u_outeredge[edge_num]
#         ii = 0
#         nn = block[0]
#
#         while nn < sza and ii < sza - 1:
#             ii += 1
#             tempblock = max(outeredge[nn] - nn, 1)  # Added max to prevent zero blocks
#             block[ii] = tempblock
#             nn = nn + tempblock
#
#         blocks.append(block)
#
#     return blocks

