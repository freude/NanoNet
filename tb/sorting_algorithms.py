import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lgmres


def sort_lexico(coords=None, **kwargs):
    return np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))


def sort_projection(coords=None, left_lead=None, right_lead=None, **kwargs):

    vec = np.mean(coords[left_lead], axis=0) - np.mean(coords[right_lead], axis=0)
    keys = np.dot(coords, vec) / np.linalg.norm(vec)

    return np.argsort(keys)


def sort_capacitance(coords, mat, left_lead, right_lead, **kwargs):
    """

    :param coords:
    :param mat:
    :param left_lead:
    :param right_lead:
    :return:
    """

    charge = np.zeros(coords.shape[0], dtype=np.complex)
    charge[left_lead] = 1e3
    charge[right_lead] = -1e3

    x = coords[:, 1].T
    y = coords[:, 0].T

    mat = (mat != 0.0).astype(np.float)
    mat = 10 * (mat - np.diag(np.diag(mat)))
    mat = np.matrix(mat - np.diag(np.sum(mat, axis=1)) + 0 * 0.01 * np.identity(mat.shape[0]))

    aaa = np.matrix(charge)
    col, info = lgmres(mat, aaa.T, x0=1.0 / np.diag(mat), tol=1e-5, maxiter=15)
    col = col / np.max(col)

    plt.scatter(x, y, c=col, cmap=plt.cm.get_cmap('gray'), s=50, marker="o", edgecolors="k")
    plt.colorbar()
    plt.show()
    indices = np.argsort(col / np.max(col), kind='heapsort')

    mat = mat[indices, :]
    mat = mat[:, indices]

    return indices
