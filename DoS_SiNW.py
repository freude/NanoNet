import numpy as np
import matplotlib.pyplot as plt
from SiNWfunction import bs
from scipy import interpolate
import os

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
    Eg, bstruct = bs(path='c:\users\sammy\desktop\NanoNet\input_samples', kk=kk, flag=False)

    E_v = np.linspace(-1, 0, 300)
    E_c = np.linspace(2, 3, 300)
    h = 0.01

    dos_c = np.zeros(E_c.shape)
    dos_v = np.zeros(E_v.shape)

    k_new = np.linspace(0, 0.57, 400, endpoint=True)

    for j in range(bstruct[0].shape[1]):
        tck = interpolate.splrep(kk, bstruct[0][:, j])
        bs_dense = interpolate.splev(k_new, tck)
        dos_c += dos(E_c, bs_dense, k_new, h)
        dos_v += dos(E_v, bs_dense, k_new, h)

    fig3, ax3 = plt.subplots()
    ax3.plot(E_c, dos_c)
    ax3.set_xlabel(r'Energy (eV)')
    ax3.set_ylabel(r'DoS')
    ax3.set_title('DoS of Conduction Band')
    fig3.tight_layout()
    plt.savefig('dos_cb.pdf', dpi=100)

    fig4, ax4 = plt.subplots()
    ax4.plot(E_v, dos_v)
    ax4.set_xlabel(r'Energy (eV)')
    ax4.set_ylabel(r'DoS')
    ax4.set_title('DoS of Valance Band')
    fig4.tight_layout()
    plt.savefig('dos_vb.pdf', dpi=100)

