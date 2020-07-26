"""
The module contains functions that computes Green's functions and their poles
"""
from __future__ import print_function, division
import pickle
import os.path
import numpy as np
from nanonet.negf.greens_functions import surface_greens_function_poles


# def object_function(vec, energy, initial_basis, extended_basis, h_0, h_0_reduced):
#
#     for eee in energy:
#
#         aaa = 1 + \
#               extended_basis.H *\
#               h_0 *\
#               initial_basis *\
#               np.pinv((z - h_0_reduced)**2) *\
#               initial_basis.H *\
#               h_0 *\
#               extended_basis
#
#         bbb = 1* z -\
#               extended_basis.H *\
#               h_0 *\
#               extended_basis - \
#               extended_basis.H * \
#               h_0 * \
#               initial_basis * \
#               np.pinv(z - h_0_reduced) * \
#               initial_basis.H * \
#               h_0 * \
#               extended_basis
#
#     return fff
def object_function1(vec, energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states):
    """

    Parameters
    ----------
    vec :
        
    energy :
        
    init_basis :
        
    extended_basis :
        
    h_l :
        
    h_0 :
        
    h_r :
        
    num_of_states :
        

    Returns
    -------

    """

    extended_basis1 = np.array(1.0 / np.sqrt(np.dot(vec, vec.conj().T))) * np.array(extended_basis * vec.T)
    extended_basis1 = np.hstack((init_basis, extended_basis1))

    h_l_reduced = np.dot(np.dot(extended_basis1.conj().T, h_l), extended_basis1)
    h_0_reduced = np.dot(np.dot(extended_basis1.conj().T, h_0), extended_basis1)
    h_r_reduced = np.dot(np.dot(extended_basis1.conj().T, h_r), extended_basis1)

    _, _, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    print(num_of_states1 - num_of_states, ' : ', vec)

    return num_of_states1 - num_of_states + (np.dot(vec, vec.conj().T) - 1.0) ** 2


def object_function(vec, energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states):
    """

    Parameters
    ----------
    vec :
        
    energy :
        
    init_basis :
        
    extended_basis :
        
    h_l :
        
    h_0 :
        
    h_r :
        
    num_of_states :
        

    Returns
    -------

    """

    extended_basis1 = np.array(1.0 / np.sqrt(np.dot(vec, vec.conj().T))) * np.array(extended_basis * vec.T)
    extended_basis1 = np.hstack((init_basis, extended_basis1))

    h_l_reduced = np.dot(np.dot(extended_basis1.conj().T, h_l), extended_basis1)
    h_0_reduced = np.dot(np.dot(extended_basis1.conj().T, h_0), extended_basis1)
    h_r_reduced = np.dot(np.dot(extended_basis1.conj().T, h_r), extended_basis1)

    _, _, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    print(num_of_states1 - num_of_states, ' : ', vec)

    return num_of_states1 - num_of_states + (np.dot(vec, vec.conj().T) - 1.0) ** 2


def bs(E, h_l, h_0, h_r):
    """

    Parameters
    ----------
    E :
        
    h_l :
        
    h_0 :
        
    h_r :
        

    Returns
    -------

    """

    vals, vects = surface_greens_function_poles(h_l, h_0 - E * np.identity(h_0.shape[0]), h_r)
    vals = np.diag(vals)

    vals_for_plot = vals.copy()

    acc = 0.001

    vals_for_plot = np.angle(vals_for_plot)
    vals_for_plot[np.abs(np.abs(vals) - 1.0) > acc] = np.nan
    inds = np.where(np.abs(np.abs(vals) - 1.0) <= acc)[0]
    vects = vects[:, inds]
    vals = np.angle(vals[inds])

    return vals, vects, vals_for_plot


def bs_vs_e(energy, h_l, h_0, h_r):
    """

    Parameters
    ----------
    energy :
        
    h_l :
        
    h_0 :
        
    h_r :
        

    Returns
    -------

    """

    init_basis = []
    vals_for_plot = []

    for E in energy:
        # print(E)
        _, vec, val_for_plot = bs(E, h_l, h_0, h_r)
        vals_for_plot.append(val_for_plot)
        if vec.size > 0:
            init_basis.append(vec)

    vals_for_plot = np.array(vals_for_plot)

    num_of_states = vals_for_plot[::4, :].size - np.count_nonzero(np.isnan(vals_for_plot[::4, :]))

    if len(init_basis) > 0:
        init_basis = np.hstack(tuple(init_basis))

    return init_basis, vals_for_plot, num_of_states


def reduce_mode_space(energy, h_l, h_0, h_r, thr, input_file=""):
    """

    Parameters
    ----------
    energy :
        
    h_l :
        
    h_0 :
        
    h_r :
        
    thr :
        
    input_file :
         (Default value = "")

    Returns
    -------

    """

    # energy = np.linspace(2.0, 3.7, 50)

    if os.path.isfile(input_file):
        input_file = os.path.dirname(input_file)

    label = '_' + "{0:.2f}".format(np.min(energy)) + '_' + "{0:.2f}".format(np.max(energy)) + '_' + str(len(energy))
    first_file = os.path.join(input_file, 'init_basis'+label+'.pkl')
    second_file = os.path.join(input_file, 'vals_for_plot'+label+'.pkl')

    if os.path.isfile(first_file) and os.path.isfile(second_file):
        # unpickle
        with open(first_file, 'rb') as infile:
            init_basis = pickle.load(infile)
        with open(second_file, 'rb') as infile:
            vals_for_plot = pickle.load(infile)

        num_of_states = vals_for_plot[::4, :].size - np.count_nonzero(np.isnan(vals_for_plot[::4, :]))
    else:
        init_basis, vals_for_plot, num_of_states = bs_vs_e(energy, h_l, h_0, h_r)
        # pickle
        with open(first_file, 'wb') as outfile:
            pickle.dump(init_basis, outfile)
        with open(second_file, 'wb') as outfile:
            pickle.dump(vals_for_plot, outfile)

    # orthogonalize initial basis

    eee, vvv = np.linalg.eig(np.dot(init_basis.conj().T, init_basis))
    init_basis = init_basis.dot(vvv).dot(np.diag(1.0/np.sqrt(eee)))
    init_basis = init_basis[:, np.where(eee > thr)[0]]

    # test reduced mode space

    h_l_reduced = np.dot(np.dot(init_basis.conj().T, h_l), init_basis)
    h_0_reduced = np.dot(np.dot(init_basis.conj().T, h_0), init_basis)
    h_r_reduced = np.dot(np.dot(init_basis.conj().T, h_r), init_basis)

    _, vals_for_plot_1, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    # while num_of_states != num_of_states1:
    #
    #     extended_basis = (1.0 - init_basis * init_basis.H) * h_0 * init_basis
    #     extended_basis1 = (1.0 - init_basis * init_basis.H) * (h_l + h_r) * init_basis
    #     extended_basis = np.matrix(np.hstack((extended_basis, extended_basis1)))
    #     eee, vvv = np.linalg.eig(extended_basis.H * extended_basis)
    #     extended_basis = extended_basis * vvv * np.matrix(np.diag(1.0 / np.sqrt(eee)))
    #     extended_basis = extended_basis[:, np.where(eee > thr)[0]]
    #     x0 = 0.5*np.ones(extended_basis.shape[1])
    #     res = minimize(object_function,
    #                    x0,
    #                    args=(energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states),
    #                    method='COBYLA', options={'maxiter': 300})
    #     vec = np.matrix(res.x)
    #
    #     extended_basis = np.matrix(np.array(1.0 / np.sqrt(vec * vec.H)) * np.array(extended_basis * vec.T))
    #     init_basis = np.hstack((init_basis, extended_basis))
    #
    #     # test reduced mode space
    #     h_l_reduced = init_basis.H * h_l * init_basis
    #     h_0_reduced = init_basis.H * h_0 * init_basis
    #     h_r_reduced = init_basis.H * h_r * init_basis
    #
    #     _, vals_for_plot2, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    return h_l_reduced, h_0_reduced, h_r_reduced, vals_for_plot, init_basis
