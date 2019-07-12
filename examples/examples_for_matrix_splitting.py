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
    h0 = scipy.sparse.random(1000, 1000, density=0.0003, random_state=100)
    h0 = h0 + np.identity(h0.shape[0]) + \
         np.diag(np.diag(np.identity(h0.shape[0]-1)), 1) +\
         np.diag(np.diag(np.identity(h0.shape[0]-1)), -1)

    h0 = h0 + h0.T
    h0_s = scipy.sparse.csc_matrix(h0)

    h0 = h0 + np.identity(h0.shape[0]) + np.diag(np.diag(np.identity(h0.shape[0] - 1)), 1) + np.diag(
        np.diag(np.identity(h0.shape[0] - 1)), -1)

    ind = scipy.sparse.csgraph.reverse_cuthill_mckee(h0_s)
    h0 = h0[:, ind]
    h0 = h0[ind, :]

    # fltr = np.triu(np.ones(h0.shape, dtype=np.bool), h0.shape[0] // 5)
    # fltr1 = np.tril(np.ones(h0.shape, dtype=np.bool), -h0.shape[0] // 5)
    # h0[fltr] = 0
    # h0[fltr1] = 0

    h0[h0.shape[0] // 2 + 10:h0.shape[0] // 2 + 10 + 200,
       h0.shape[0] // 2 - 10 - 200: h0.shape[0] // 2 - 10] = 0

    h0[h0.shape[0] // 2 - 10 - 200:h0.shape[0] // 2 - 10,
       h0.shape[0] // 2 + 10:h0.shape[0] // 2 + 10 + 200] = 0

    h0[h0.shape[0] // 3 + 10:h0.shape[0] // 3 + 10 + 200,
       h0.shape[0] // 3 - 10 - 200: h0.shape[0] // 3 - 10] = 0

    h0[h0.shape[0] // 3 - 10 - 200:h0.shape[0] // 3 - 10,
       h0.shape[0] // 3 + 10:h0.shape[0] // 3 + 10 + 200] = 0

    h0[h0.shape[0] // 4 + 10:h0.shape[0] // 4 + 10 + 200,
       h0.shape[0] // 4 - 10 - 200: h0.shape[0] // 4 - 10] = 0

    h0[h0.shape[0] // 4 - 10 - 200:h0.shape[0] // 4 - 10,
       h0.shape[0] // 4 + 10:h0.shape[0] // 4 + 10 + 200] = 0

    h0[h0.shape[0] // 5 + 10:h0.shape[0] // 5 + 10 + 200,
       h0.shape[0] // 5 - 10 - 200: h0.shape[0] // 5 - 10] = 0

    h0[h0.shape[0] // 5 - 10 - 200:h0.shape[0] // 5 - 10,
       h0.shape[0] // 5 + 10:h0.shape[0] // 5 + 10 + 200] = 0

    h01, hl1, hr1, subblocks = split_matrix(h0)
    # h01, hl1, hr1, subblocks = split_into_subblocks(h0, 50, 50)
    # print(subblocks)

    hr = h0
    hl = h0

    a = np.block([[h0, hr], [hl, h0]])
    # h01, hl1, hr1, subblocks = split_into_subblocks(h0, h_l=hl, h_r=hr)
    # print(len(h01))

    b = np.zeros(a.shape, np.complex)

    j1 = 0
    for j in range(1):
        for num, item in enumerate(h01):
            b[j1:j1 + item.shape[0], j1:j1 + item.shape[1]] = item
            if num < len(h01) - 1:
                b[j1:j1 + item.shape[0],
                  j1 + item.shape[1]:j1 + item.shape[1] + h01[num + 1].shape[1]] = hr1[num]

                b[j1 + item.shape[0]:j1 + item.shape[0] + h01[num + 1].shape[0],
                  j1:j1 + item.shape[1]] = hl1[num]

            # if num == len(h01) - 1 and j == 0:
            #     b[:j1 + item.shape[0], j1 + item.shape[1]:] = hr
            #     b[j1 + item.shape[0]:, :j1 + item.shape[1]] = hl

            j1 += item.shape[0]

    cumsum = np.cumsum(np.array(subblocks))[:-1]
    cumsum = np.insert(cumsum, 0, 0)

    fig, ax = plt.subplots(1)

    b = b[:b.shape[0]//2, :b.shape[0]//2]

    ax.spy(np.abs(b))

    ind0 = np.arange(b.shape[0])
    ind_bl = np.arange(len(h01))

    def nested_dissection_1d(array):

        if np.size(array) > 2:

            array1 = array[:np.size(array) // 2]
            array2 = array[np.size(array) // 2]
            array3 = array[np.size(array) // 2 + 1:]

            # array1 = array[:np.size(array) // 3]
            # array2 = array[np.size(array) // 3 : 2* np.size(array) // 3]
            # array3 = array[2* np.size(array) // 3:]

            ans1 = nested_dissection_1d(array1)
            ans2 = nested_dissection_1d(array3)

            return np.concatenate((ans1, ans2, array2), axis=None)
            # return np.concatenate((ans1, ans3, ans2), axis=None)

        else:

            return array


    def nested_dissection_1d_v2(array, couplings):

        if np.size(array) > 2:

            ind = np.size(array) // 2
            print(np.argmax(couplings[ind-1:ind+2]))
            ind = ind - 1 + np.argmax(couplings[ind-1:ind+2])

            array1 = array[:ind]
            couplings1 = couplings[:ind]
            array2 = array[ind]
            array3 = array[ind + 1:]
            couplings3 = couplings[ind + 1:]

            # array1 = array[:np.size(array) // 3]
            # array2 = array[np.size(array) // 3 : 2* np.size(array) // 3]
            # array3 = array[2* np.size(array) // 3:]

            ans1 = nested_dissection_1d_v2(array1, couplings1)
            ans2 = nested_dissection_1d_v2(array3, couplings3)

            return np.concatenate((ans1, ans2, array2), axis=None)
            # return np.concatenate((ans1, ans3, ans2), axis=None)

        else:

            return array

    ind = []
    counter = 0

    for num, item in enumerate(h01):
        ind.append(ind0[counter:counter+item.shape[0]])
        counter += item.shape[0]

    ind = np.array(ind)

    sizes1 = [0] + [np.count_nonzero(item) for item in hr1]
    sizes2 = [np.count_nonzero(item) for item in hl1] + [0]
    sizes = np.array(sizes1) + np.array(sizes2)
    sizes = [item.shape[0] for item in h01]
    # sizes = 1.0 / np.array(sizes0)
    # ind_bl = [x for _, x in sorted(zip(sizes, ind_bl))]
    # ind_bl = ind_bl[::-1]

    # ind1 = ind_bl[ind_bl % 2 == 0]
    # ind2 = ind_bl[ind_bl % 2 == 1]
    # ind_bl = np.concatenate((ind2, ind1))
    #
    # ind_bl = nested_dissection_1d(ind_bl)
    ind_bl = nested_dissection_1d_v2(ind_bl, sizes)
    # ind_bl = [item for item in ind_bl if item is not None]
    #
    # ind1 = ind_bl[:len(ind_bl) // 2]
    # ind2 = ind_bl[len(ind_bl) // 2:]
    # ind_bl = np.concatenate((ind1, ind2[::-1]))
    #
    # # ind_bl = np.array([1, 3, 2, 7, 5, 6, 4]) - 1
    # ind = ind[ind_bl]

    # ind = nested_dissection_1d(np.arange(b.shape[0]))
    # ind = np.concatenate(ind)

    # ind1 = ind_bl[ind_bl % 2 == 0]
    # ind2 = ind_bl[ind_bl % 2 == 1]
    #
    # ind_bl = np.concatenate((ind2, ind1))
    ind = ind[ind_bl]
    ind = np.concatenate(ind)

    b = b[:, ind]
    b = b[ind, :]

    from scipy.linalg import lu

    p, l, u = lu(b)
    b1 = np.matrix(l) * np.matrix(u)

    for jj in range(1):
        cumsum = cumsum + jj * h0.shape[0]

        if jj == 1:
            rect = Rectangle((h0.shape[0] - h01[-1].shape[0] - 0.5, h0.shape[1] - 0.5), h01[-1].shape[1], h01[0].shape[0],
                             linestyle='--',
                             linewidth=1,
                             edgecolor='b',
                             facecolor='none')
            ax.add_patch(rect)
            rect = Rectangle((h0.shape[0] - 0.5, h0.shape[1] - h01[-1].shape[1] - 0.5), h01[0].shape[1], h01[-1].shape[0],
                             linestyle='--',
                             linewidth=1,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        for j, item in enumerate(cumsum):
            if j < len(cumsum) - 1:
                rect = Rectangle((item - 0.5, cumsum[j + 1] - 0.5), subblocks[j], subblocks[j + 1],
                                 linewidth=1,
                                 edgecolor='b',
                                 facecolor='none')
                ax.add_patch(rect)
                rect = Rectangle((cumsum[j + 1] - 0.5, item - 0.5), subblocks[j + 1], subblocks[j],
                                 linewidth=1,
                                 edgecolor='g',
                                 facecolor='none')
                ax.add_patch(rect)
            rect = Rectangle((item - 0.5, item -0.5), subblocks[j], subblocks[j],
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)

    # plt.xlim(b.shape[0] / 2 - 0.5, -1.0)
    # plt.ylim(-1.0, b.shape[0] / 2 - 0.5 )
    plt.axis('off')
    plt.show()
