#!/usr/bin/env python
from __future__ import print_function
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tb


def main():

    def main1(param_file, k_points_file, xyz, show, save):

        params = tb.yaml_parser(param_file)    # parse parameter file

        if k_points_file is None:   # default k-points
            if len(params['primitive_cell']) == 3:
                sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
                num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]
                wave_vector = tb.get_k_coords(sym_points, num_points)
            else:
                wave_vector = [[0.0, 0.0, 0.0]]
        else:   # read k-points file if its path is provided from the command line arguments
            wave_vector = np.loadtxt(k_points_file)

        # read xyz file if its path is provided from the command line arguments
        if isinstance(xyz, str):
            with open(xyz, 'r') as myfile:
                params['xyz'] = myfile.read()

        # initialize Hamiltonian
        hamiltonian = tb.initializer(**params)

        # compute band structure
        band_structure = []

        band_structure = [{} for _ in xrange(len(wave_vector))]

        for j, jj in enumerate(wave_vector):
            vals, vects = hamiltonian.diagonalize_periodic_bc(jj)
            band_structure[j] = {'wave_vector': jj, 'eigenvalues': vals, 'eigenvectors': vects}
            print('#{} '.format(j), " ".join(['{:.3f} '.format(element) for element in vals]))

        band_structure = np.array([band_structure[item]['eigenvalues']
                                   for item in xrange(len(band_structure))])

        if show:

            axes = plt.axes()
            axes.set_ylim(-1.0, 2.7)
            axes.set_title('Band structure')
            axes.set_xlabel('Wave vectors')
            axes.set_ylabel('Energy (eV)')
            axes.plot(band_structure)

            if show != 2:
                plt.show()
            else:
                plt.savefig('band_structure.png')

        if save:
            with open('./band_structure.pkl', 'wb') as f:
                pickle.dump(band_structure, f, pickle.HIGHEST_PROTOCOL)

    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str,
                        help='Path to the yaml file containing parameters')

    parser.add_argument('--k_points_file', type=str, default=None,
                        help='Path to the file containing k points coordinates')

    parser.add_argument('--xyz', type=str, default=None,
                        help='Path to the file containing atomic coordinates')

    parser.add_argument('--show', '-S', type=int, default=1,
                        help='Show figures, 0/1')

    parser.add_argument('--save', '-s', type=int, default=1,
                        help='Save results of computations on disk, 0/1')

    args = parser.parse_args()
    main1(args.param_file, args.k_points_file, args.xyz, args.show, args.save)


if __name__ == '__main__':

    main()
