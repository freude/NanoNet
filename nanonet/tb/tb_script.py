#!/usr/bin/env python
from __future__ import print_function
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb


def preprocess_data(param_file, k_points_file, xyz, code_name):
    """

    Parameters
    ----------
    param_file :
        
    k_points_file :
        
    xyz :
        
    code_name :
        

    Returns
    -------

    """
    params = tb.yaml_parser(param_file)    # parse parameter file

    if k_points_file is None:   # default k-points
        if len(params['primitive_cell']) == 3:
            sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
            num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]
            wave_vector = tb.get_k_coords(sym_points, num_points, 'Si')
        else:
            wave_vector = [[0.0, 0.0, 0.0]]
    else:   # read k-points file if its path is provided from the command line arguments
        wave_vector = np.loadtxt(k_points_file)

    # read xyz file if its path is provided from the command line arguments
    if isinstance(xyz, str):
        with open(xyz, 'r') as myfile:
            params['xyz'] = myfile.read()

    if not isinstance(code_name, str):
        code_name = 'band_structure'

    return params, wave_vector, code_name


def postprocess_data(kk, band_structure, show, save, code_name):
    """

    Parameters
    ----------
    kk :
        
    band_structure :
        
    show :
        
    save :
        
    code_name :
        

    Returns
    -------

    """

    flag = True  # 1D system

    if not isinstance(band_structure, np.ndarray):
        ids = [band_structure[item]['id'] for item in range(len(band_structure))]
        band_structure = [x['eigenvalues'] for _, x in sorted(zip(ids, band_structure))]
        band_structure = np.array(band_structure)

    if len(kk.shape) > 1:
        kkk = np.sum(kk, axis=0)
        if np.count_nonzero(kkk) == 1:
            kk = kk[:, np.where(kkk != 0.0)[0][0]]
        else:
            kk = list(range(kk.shape[0]))
            flag = False

    if show:

        vb = np.sort(np.real(band_structure))
        cb = vb.copy()
        vb[vb > 0] = np.NaN
        cb[cb < 0] = np.NaN

        vb = -np.sort(-vb)
        cb = np.sort(cb)

        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            if flag:
                fig.set_figheight(7)
                fig.set_figwidth(4)
                ax_lim1 = np.nanmax(vb) - 0.5
                ax_lim2 = np.nanmax(vb) + 0.1
                ax.set_ylim(ax_lim1, ax_lim2)
            ax.plot(kk, vb, marker="o", markersize=5)
            ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
            ax.set_ylabel(r'Energy (eV)')
            ax.set_title('Valence band')
            fig.tight_layout()
            if show != 2:
                plt.show()
            else:
                plt.savefig(code_name + '_vb.pdf')
        except ValueError:
            pass

        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            if flag:
                fig.set_figheight(7)
                fig.set_figwidth(4)
                ax_lim1 = np.nanmin(cb) - 0.1
                ax_lim2 = np.nanmin(cb) + 0.7
                ax.set_ylim(ax_lim1, ax_lim2)
            ax.plot(kk, cb, marker="o", markersize=5)
            ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
            ax.set_ylabel(r'Energy (eV)')
            ax.set_title('Conduction band')
            fig.tight_layout()
            if show != 2:
                plt.show()
            else:
                plt.savefig(code_name + '_cb.pdf')
        except ValueError:
            pass

    if save:
        with open(code_name+'.pkl', 'wb') as f:
            pickle.dump(band_structure, f, pickle.HIGHEST_PROTOCOL)


def main1(param_file, k_points_file, xyz, show, save, code_name):
    """

    Parameters
    ----------
    param_file :
        
    k_points_file :
        
    xyz :
        
    show :
        
    save :
        
    code_name :
        

    Returns
    -------

    """

    params, wave_vector, code_name = preprocess_data(param_file, k_points_file, xyz, code_name)

    # initialize Hamiltonian
    hamiltonian = tb.initializer(**params)

    # compute band structure
    band_structure = [{} for _ in range(len(wave_vector))]

    for j, jj in enumerate(wave_vector):
        vals, vects = hamiltonian.diagonalize_periodic_bc(jj)
        band_structure[j] = {'id': j, 'wave_vector': jj, 'eigenvalues': vals, 'eigenvectors': vects}
        print('#{} '.format(j), " ".join(['{:.3f} '.format(element) for element in vals]))

    postprocess_data(wave_vector, band_structure, show, save, code_name)

    return 0


def create_parser():
    """ """

    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str,
                        help='Path to the file in the yaml-format\
                             containing all parameters needed to run computations.')

    parser.add_argument('--k_points_file', type=str, default=None,
                        help='Path to the txt file containing coordinates of \
                             wave vectors for the band structure computations.\
                             If not specified, default values will be used.')

    parser.add_argument('--xyz', type=str, default=None,
                        help='Path to the file containing atomic coordinates. \
                             If specified, it overrides the coordinates \
                             specified in the param_files.')

    parser.add_argument('--show', '-S', type=int, default=1,
                        help='Show figures, 0/1/2. \
                             0 shows nothing,  \
                             1 outputs figures on screen, \
                             2 saves figures on disk without showing.')

    parser.add_argument('--save', '-s', type=int, default=0,
                        help='Save results of computations on disk, 0/1.')

    parser.add_argument('--code_name', type=str, default=None,
                        help='Code name is added to the names of all saved data files.')

    return parser


def main():
    """ """

    parser = create_parser()
    args = parser.parse_args()
    main1(args.param_file, args.k_points_file, args.xyz, args.show, args.save, args.code_name)

    return 0


if __name__ == '__main__':

    main()
