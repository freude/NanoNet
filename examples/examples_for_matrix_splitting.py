import numpy as np
from tb import Orbitals, Hamiltonian
from tb.aux_functions import split_into_subblocks, split_matrix


if __name__ == "__main__":

    # a = np.array([[1, 1], [1, 1]])
    # b = np.zeros((2, 2))
    # test_matrix = np.block([[a, b, b], [b, a, b], [b, b, a]])
    # test_matrix[0, 2] = 2
    # test_matrix[2, 0] = 2
    # test_matrix[2, 4] = 3
    # test_matrix[4, 2] = 3
    #
    # a, b = compute_edge(test_matrix)
    # ans = blocksandborders_constrained(0, 0, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(1, 1, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(2, 2, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(3, 3, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(4, 4, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(5, 5, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(2, 4, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(2, 3, a, b)
    # print(ans)
    # ans = blocksandborders_constrained(3, 2, a, b)
    # print(ans)


    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    # band_gaps = []
    # band_structures = []
    #
    # path = "./input_samples/SiNW2.xyz"
    #
    # hamiltonian = Hamiltonian(xyz=path, nn_distance=2.4, so_coupling=0.06, vec=[0, 0, 1])
    # hamiltonian.initialize()
    #
    # # if True:
    # #     plt.axis('off')
    # #     plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
    # #     plt.savefig('hamiltonian.pdf')
    # #     plt.show()
    #
    # a_si = 5.50
    # PRIMITIVE_CELL = [[0, 0, a_si]]
    # hamiltonian.set_periodic_bc(PRIMITIVE_CELL)
    #
    # hl, h0, hr = hamiltonian.get_coupling_hamiltonians()

    import scipy
    h0 = scipy.sparse.random(500, 500, density=0.0005)
    h0 = h0 + np.identity(h0.shape[0]) + np.diag(np.diag(np.identity(h0.shape[0]-1)), 1) + np.diag(np.diag(np.identity(h0.shape[0]-1)), -1)
    h0 = h0 + h0.T
    h0_s = scipy.sparse.csc_matrix(h0)
    ind = scipy.sparse.csgraph.reverse_cuthill_mckee(h0_s)
    h0 = h0[:, ind]
    h0 = h0[ind, :]

    h01, hl1, hr1, subblocks = split_matrix(h0)
    print(subblocks)

    hr = h0
    hl = h0

    a = np.block([[h0, hr], [hl, h0]])
    # h01, hl1, hr1, subblocks = split_into_subblocks(h0, h_l=hl, h_r=hr)
    # print(len(h01))

    b = np.zeros(a.shape, np.complex)

    j1 = 0
    for j in range(2):
        for num, item in enumerate(h01):
            b[j1:j1 + item.shape[0], j1:j1 + item.shape[1]] = item
            if num < len(h01) - 1:
                b[j1:j1 + item.shape[0],
                  j1 + item.shape[1]:j1 + item.shape[1] + h01[num + 1].shape[1]] = hr1[num]

                b[j1 + item.shape[0]:j1 + item.shape[0] + h01[num + 1].shape[0],
                  j1:j1 + item.shape[1]] = hl1[num]

            if num == len(h01) - 1 and j == 0:
                b[:j1 + item.shape[0], j1 + item.shape[1]:] = hr
                b[j1 + item.shape[0]:, :j1 + item.shape[1]] = hl

            j1 += item.shape[0]

    cumsum = np.cumsum(np.array(subblocks))[:-1]
    cumsum = np.insert(cumsum, 0, 0)

    fig, ax = plt.subplots(1)

    ax.spy(np.abs(b))

    for jj in range(1):
        cumsum = cumsum + jj * h0.shape[0]

        if jj == 1:
            rect = Rectangle((h0.shape[0] - h01[-1].shape[0], h0.shape[1]), h01[-1].shape[1], h01[0].shape[0],
                             linestyle='--',
                             linewidth=1,
                             edgecolor='b',
                             facecolor='none')
            ax.add_patch(rect)
            rect = Rectangle((h0.shape[0], h0.shape[1] - h01[-1].shape[1]), h01[0].shape[1], h01[-1].shape[0],
                             linestyle='--',
                             linewidth=1,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        for j, item in enumerate(cumsum):
            if j < len(cumsum) - 1:
                rect = Rectangle((item, cumsum[j + 1]), subblocks[j], subblocks[j + 1],
                                 linewidth=1,
                                 edgecolor='b',
                                 facecolor='none')
                ax.add_patch(rect)
                rect = Rectangle((cumsum[j + 1], item), subblocks[j + 1], subblocks[j],
                                 linewidth=1,
                                 edgecolor='g',
                                 facecolor='none')
                ax.add_patch(rect)
            rect = Rectangle((item, item), subblocks[j], subblocks[j],
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)

    plt.xlim(b.shape[0] / 2 + 0.5, -0.5)
    plt.ylim(-0.5, b.shape[0] / 2 + 0.5 )
    plt.axis('off')
    plt.show()