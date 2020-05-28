import numpy as np
import matplotlib.pyplot as plt
import nanonet.tb as tb
from nanonet.tb.reduced_mode_space import bs, reduce_mode_space
import pickle


if __name__ == '__main__':

    # make hamiltonian matrices
    tb.Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    h = tb.Hamiltonian(xyz='/home/mk/TB_project/input_samples/SiNW2.xyz', nn_distance=2.4)
    h.initialize()
    period = [0, 0, 5.50]
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_hamiltonians()

    # compute band structure
    energy = np.linspace(1.4, 1.9, 30)

    h_l_reduced, h_0_reduced, h_r_reduced, vals_for_plot, a = reduce_mode_space(energy, h_l, h_0, h_r, 0.1)

    # vals = []
    # vecs = []
    # vals_for_plot = []
    #
    # for E in energy:
    #     print(E)
    #     val, vec, val_for_plot = bs(E, h_l, h_0, h_r)
    #     vals.append(val)
    #     vecs.append(vec)
    #     vals_for_plot.append(val_for_plot)
    #
    # outfile = open('vals.pkl', 'wb')
    # pickle.dump(vals, outfile)
    # outfile.close()
    # outfile = open('vecs.pkl', 'wb')
    # pickle.dump(vecs, outfile)
    # outfile.close()
    # outfile = open('vals_for_plot.pkl', 'wb')
    # pickle.dump(vals_for_plot, outfile)
    # outfile.close()
    #
    # infile = open('vals.pkl', 'rb')
    # vals = pickle.load(infile)
    # infile.close()
    # infile = open('vecs.pkl', 'rb')
    # vecs = pickle.load(infile)
    # infile.close()
    # infile = open('vals_for_plot.pkl', 'rb')
    # vals_for_plot = pickle.load(infile)
    # infile.close()

    plt.plot(energy, vals_for_plot, 'o', fillstyle='none')
    # orthogonalize initial basis

    vals = []
    vecs = []
    vals_for_plot1 = []

    energy1 = np.linspace(1.4, 1.9, 200)
    # energy1 = np.linspace(2.0, 3.7, 50)

    for E in energy1:
        print(E)
        val, vec, val_for_plot = bs(E, h_l_reduced, h_0_reduced, h_r_reduced)
        vals.append(val)
        vecs.append(vec)
        vals_for_plot1.append(val_for_plot)

    vals_for_plot1 = np.array(vals_for_plot1)
    plt.plot(energy1, vals_for_plot1, '.')
    plt.show()
    print('hi')
