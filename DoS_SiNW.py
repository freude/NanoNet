import interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
from SiNWfunction import bs
from scipy.interpolate import *
import scipy as sp
from scipy import interpolate


def delta(E, eps, h):
    if E - h < eps < E + h:
        return 1
    else:
        return 0


def dos(E, bs, kk, h):

    dos = np.zeros(E.shape)
    for j, en in enumerate(E):
        for j1 in range(len(kk)):
            dos[j] += delta(en, bs[j1], h)
    return dos


if __name__ == '__main__':

    num_points = 20
    kk = np.linspace(0, 0.57, num_points, endpoint=True)
    Eg, bstruct = bs(path='c:\users\sammy\desktop\NanoNet\input_samples', kk=kk, flag=True)

    E = np.linspace(-2, 0, 200)
    h = 0.01

    dos1 = np.zeros(E.shape)

    tck = interpolate.splrep(bstruct, num_points)
    k_new = np.arange(0, num_points, 5)
    bs_dense = interpolate.splrep(k_new, tck)

    for j in range(bstruct[0].shape[1]):
        bs_dense = bstruct[0][:, j]
        dos1 += dos(E, bs_dense, k_new, h)
    print(dos1)
