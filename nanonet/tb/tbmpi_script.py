#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from mpi4py import MPI
import nanonet.tb as tb
from .tb_script import create_parser, preprocess_data, postprocess_data


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    band_structure = []

    for j, jj in enumerate(wave_vector):
        if j % size != rank:
            continue

        vals, vects = hamiltonian.diagonalize_periodic_bc(jj)
        band_structure.append({'id': j, 'wave_vector': jj, 'eigenvalues': vals, 'eigenvectors': vects})
        print('#{} '.format(j), " ".join(['{:.3f} '.format(element) for element in vals]))

    band_structure = comm.reduce(band_structure, root=0)

    if rank == 0:
        postprocess_data(wave_vector, band_structure, show, save, code_name)

    return 0


def main():
    """ """

    parser = create_parser()
    args = parser.parse_args()
    main1(args.param_file, args.k_points_file, args.xyz, args.show, args.save, args.code_name)

    return 0


if __name__ == '__main__':

    main()
