#!/usr/bin/env python
from __future__ import print_function
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nanonet.tb as tb


def main():

    def main(param_file, energy, show, save):

        params = tb.yaml_parser(param_file)
        hamiltonian = tb.initializer(**params)
        h_l, h_0, h_r = hamiltonian.get_hamiltonians()

        sgf_l = []
        sgf_r = []

        for E in energy:
            L, R = tb.surface_greens_function(E, h_l, h_0, h_r)
            sgf_l.append(L)
            sgf_r.append(R)

        sgf_l = np.array(sgf_l)
        sgf_r = np.array(sgf_r)

        num_sites = h_0.shape[0]
        gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

        dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

        tr = np.zeros((energy.shape[0]), dtype=np.complex)

        for j, E in enumerate(energy):
            gf0 = gf[j, :, :]
            gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
            gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
            tr[j] = np.real(np.trace(gamma_l * gf0 * gamma_r * gf0.conj().T))
            dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))

        if show:
            axes = plt.axes()
            axes.set_title('Band structure')
            axes.set_xlabel('Wave vectors')
            axes.set_ylabel('Energy (eV)')
            axes.plot(dos)
            plt.show()

        if save:
            pass
            # with open('./band_structure.pkl', 'wb') as f:
            #     pickle.dump(band_structure, f, pickle.HIGHEST_PROTOCOL)


    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str,
                        help='Path to the yaml file containing parameters')

    parser.add_argument('--k_points_file', type=str, default=None,
                        help='Path to the file containing k points coordinates')

    parser.add_argument('--show', '-S', type=int, default=1,
                        help='Show figures, 0/1')

    parser.add_argument('--save', '-s', type=int, default=1,
                        help='Save results of computations on disk, 0/1')

    args = parser.parse_args()
    main(args.param_file, args.k_points_file, args.show, args.save)
