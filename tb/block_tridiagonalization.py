import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
import math
import scipy


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


def cut_in_blocks(h_0, blocks):
    """
    Cut a matrix into diagonal, upper-diagonal and lower-diagonal blocks
    if sizes of the diagonal blocks are specified.

    Parameters
    ----------
    h_0 : ndarray
        Input matrix
    blocks : ndarray(dtype=int)
        Sizes of diagonal blocks

    Returns
    -------
    h_0_s, h_l_s, h_r_s : ndarray
        List of diagonal matrices,
        list of lower-diagonal matrices and
        list of upper-diagonal matrices.

        Note that if the size of the list h_0_s is N,
        the sizes of h_l_s, h_r_s are N-1.

    Examples
    --------
    >>> import numpy as np
    >>> from tb.block_tridiagonalization import cut_in_blocks
    >>> a = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    >>> a
    array([[1, 1, 0, 0],
           [1, 1, 1, 0],
           [0, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> # Sum the diagonals.
    >>> blocks = [2, 2]
    >>> blocks
    [2, 2]
    >>> h0, h1, h2 = cut_in_blocks(a, blocks)
    >>> h0
    [array([[1, 1],
           [1, 1]]), array([[1, 1],
           [1, 1]])]
    >>> h1
    [array([[0, 1],
           [0, 0]])]
    >>> h2
    [array([[0, 0],
           [1, 0]])]
    """

    j1 = 0

    h_0_s = []
    h_l_s = []
    h_r_s = []

    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block

    return h_0_s, h_l_s, h_r_s


def find_optimal_cut(edge, edge1, left, right):
    """
    Computes the index corresponding to the optimal cut such that applying
    the function compute_blocks() to the subblocks defined by the cut reduces
    the cost function comparing to the case when the function compute_blocks() is
    applied to the whole matrix. If cutting point can not be find, the algorithm returns
    the result from the function compute_blocks().

    Parameters
    ----------
    edge : ndarray
        sparsity pattern profile of the matrix
    edge1 : ndarray
        conjugated sparsity pattern profile of the matrix
    left : int
        size of the leftmost diagonal block
    right : int
        size of the rightmost diagonal block

    Returns
    -------
    blocks : list
        list of diaginal block sizes

    sep : int
        the index of the optimal cut

    right_block : int
        size of the rightmost subblock of the left block (relative to the cutting point)

    left_block : int
        size of the leftmost subblock of the right block (relative to the cutting point)

    """

    unique_indices = np.arange(left, len(edge) - right + 1)
    blocks = []
    seps = []
    sizes = []
    metric = []
    size = len(edge)

    for j1, item1 in enumerate(unique_indices):

        seps.append(item1)
        item2 = size-item1

        # print(item1, item2)
        # print(item1)

        edge_1 = edge[:item1]
        edge_2 = (edge1-np.arange(len(edge1)))[item2:] + np.arange(item1)

        edge_3 = edge1[:item2]
        edge_4 = (edge-np.arange(len(edge)))[item1:] + np.arange(item2)

        block1 = compute_blocks(left, (edge1 - np.arange(len(edge)))[item2],
                                edge_1, edge_2)

        block2 = compute_blocks(right, (edge - np.arange(len(edge1)))[item1],
                                edge_3, edge_4)

        block = block1 + block2[::-1]
        blocks.append(block)
        metric.append(np.sum(np.array(block) ** 3))
        sizes.append((block1[-1], block2[-1]))

    if len(metric) == 0:
        return [left, right], np.nan, 0, 0
    else:

        best = np.argmin(np.array(metric))

        blocks = blocks[best]
        blocks = [item for item in blocks if item != 0]

        sep = seps[best]

        right_block, left_block = sizes[best]

        return blocks, sep, right_block, left_block


def compute_blocks_optimized(edge, edge1, left=1, right=1):
    """
    Computes optimal sizes of diagonal blocks of a matrix whose
    sparsity pattern is defined by the sparsity pattern profiles edge and edge1.
    This function is based on the algorithm which uses defined above function
    find_optimal_cut() to subdivide the problem into subproblems in a optimal way
    according to some cost function.

    Parameters
    ----------
    edge : ndarray
        sparsity pattern profile of the matrix
    edge1 : ndarray
        conjugated sparsity pattern profile of the matrix
    left : int
        size of the leftmost diagonal block (constrained)
    right : int
        size of the rightmost diagonal block (constrained)

    Returns
    -------
    blocks : list
        list of optimal sizes of diagonal blocks

    """

    blocks, sep, right_block, left_block = find_optimal_cut(edge, edge1, left=left, right=right)
    flag = False

    if not math.isnan(sep):

        # print(left, right_block, sep)

        if left + right_block < sep:

            edge_1 = edge[:sep]
            # edge_1[edge_1 > sep] = sep
            edge_2 = (edge1-np.arange(len(edge1)))[-sep:] + np.arange(sep)

            blocks1 = compute_blocks_optimized(edge_1, edge_2, left=left, right=right_block)

        elif left + right_block == sep:

            blocks1 = [left, right_block]
        else:

            flag = True

        # print(left_block, right, len(edge) - sep)

        if right + left_block < len(edge) - sep:

            edge_3 = (edge-np.arange(len(edge)))[sep:] + np.arange(len(edge) - sep)
            edge_4 = edge1[:-sep]
            # edge_4[edge_4 > len(edge) - sep] = len(edge) - sep

            blocks2 = compute_blocks_optimized(edge_3, edge_4, left=left_block, right=right)

        elif right + left_block == len(edge) - sep:
            blocks2 = [left_block, right]
        else:
            flag=True

        if flag:
            return blocks
        else:
            blocks = blocks1 + blocks2

            return blocks


def split_into_subblocks_optimized(h_0, left=1, right=1):
    """

    :param h_0:
    :param left:
    :param right:
    :return:
    """

    if left + right > h_0.shape[0]:
        return [h_0.shape[0]]
    else:
        edge, edge1 = compute_edge(h_0)
        return compute_blocks_optimized(edge, edge1, left=left, right=right)


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

    if isinstance(h_l, np.ndarray) and isinstance(h_r, np.ndarray):
        h_r_h = find_nonzero_lines(h_r, 'bottom')
        h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
        h_l_h = find_nonzero_lines(h_l, 'top')
        h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')

    if isinstance(h_l, int) and isinstance(h_r, int):
        h_l_h = h_l
        h_r_v = h_l
        h_r_h = h_r
        h_l_v = h_r

    edge, edge1 = compute_edge(h_0)

    left_block = max(h_l_h, h_r_v)
    right_block = max(h_r_h, h_l_v)

    blocks = compute_blocks(left_block, right_block, edge, edge1)
    j1 = 0

    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block

    return h_0_s, h_l_s, h_r_s, blocks


def compute_edge(mat):
    """

    :param mat:
    :return:
    """
    # First get some statistics
    if isinstance(mat, scipy.sparse.lil.lil_matrix):
        row, col = mat.nonzero()
    else:
        row, col = np.where(mat != 0.0)  # Output rows and columns of all non-zero elements.

    # Clever use of accumarray:
    outeredge = accum(row, col, np.max) + 1
    outeredge[0] = max(0, outeredge[0])
    outeredge = np.maximum.accumulate(outeredge)

    outeredge1 = accum(np.max(row) - row[::-1], np.max(row) - col[::-1], np.max) + 1
    outeredge1[0] = max(0, outeredge1[0])
    outeredge1 = np.maximum.accumulate(outeredge1)

    return outeredge, outeredge1


def compute_blocks(left_block, right_block, edge, edge1):
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

            blocks = compute_blocks(new_left_block,
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


def show_blocks(subblocks, input_mat):

    cumsum = np.cumsum(np.array(subblocks))[:-1]
    cumsum = np.insert(cumsum, 0, 0)

    fig, ax = plt.subplots(1)
    plt.spy(input_mat, marker='.', markersize=0.9, c='k')
    # plt.plot(edge)

    for jj in range(2):
        cumsum = cumsum + jj * input_mat.shape[0]

        if jj == 1:
            rect = Rectangle((input_mat.shape[0] - subblocks[-1] - 0.5, input_mat.shape[1] - 0.5),
                             subblocks[-1], subblocks[0],
                             linestyle='--',
                             linewidth=1.3,
                             edgecolor='b',
                             facecolor='none')
            ax.add_patch(rect)
            rect = Rectangle((input_mat.shape[0] - 0.5, input_mat.shape[1] - subblocks[-1] - 0.5),
                             subblocks[0], subblocks[-1],
                             linestyle='--',
                             linewidth=1.3,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        for j, item in enumerate(cumsum):
            if j < len(cumsum) - 1:
                rect = Rectangle((item - 0.5, cumsum[j + 1] - 0.5), subblocks[j], subblocks[j + 1],
                                 linewidth=1.3,
                                 edgecolor='b',
                                 facecolor='none')
                ax.add_patch(rect)
                rect = Rectangle((cumsum[j + 1] - 0.5, item - 0.5), subblocks[j + 1], subblocks[j],
                                 linewidth=1.3,
                                 edgecolor='g',
                                 facecolor='none')
                ax.add_patch(rect)
            rect = Rectangle((item - 0.5, item -0.5), subblocks[j], subblocks[j],
                             linewidth=1.3,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)

    plt.xlim(input_mat.shape[0] - 0.5, -1.0)
    plt.ylim(-1.0, input_mat.shape[0] - 0.5)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()