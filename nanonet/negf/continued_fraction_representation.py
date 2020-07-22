import numpy as np
from scipy.linalg import eig
from nanonet.negf.hamiltonian_chain import fd


def t_order_frac(x):
    """

    Parameters
    ----------
    x :
        

    Returns
    -------

    """
    return 0.5 * (np.sign(x) + 1.0) / x


def approximant(energy, poles, residues):
    """

    Parameters
    ----------
    energy :
        
    poles :
        
    residues :
        

    Returns
    -------

    """

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = ans - residues[j] / (arg - 1j / poles[j]) - residues[j] / (arg + 1j / poles[j])

    return ans


def approximant_diff(energy, poles, residues):
    """

    Parameters
    ----------
    energy :
        
    poles :
        
    residues :
        

    Returns
    -------

    """

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5 * 0

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = ans - residues[j] / (arg - 1j / poles[j]) ** 2 - residues[j] / (arg + 1j / poles[j]) ** 2

    return ans


def poles_and_residues(cutoff=2):
    """

    Parameters
    ----------
    cutoff :
         (Default value = 2)

    Returns
    -------

    """

    b_mat = [1 / (2.0 * np.sqrt((2*(j+1) - 1)*(2*(j+1) + 1))) for j in range(0, cutoff-1)]
    b_mat = np.diag(b_mat, -1) + np.diag(b_mat, 1)

    poles, residues = eig(b_mat)

    # arg = np.argmax(np.abs(residues), axis=0)

    residues = 0.25 * np.array([np.abs(residues[0, j])**2 / (poles[j] ** 2) for j in range(residues.shape[0])])

    return poles, residues


def test_gf1(z):
    """

    Parameters
    ----------
    z :
        

    Returns
    -------

    """
    return t_order_frac(z + 10.0) + \
           t_order_frac(z + 5.0) + \
           t_order_frac(z + 2.0) + \
           t_order_frac(z - 5.0)


def test_gf(z):
    """

    Parameters
    ----------
    z :
        

    Returns
    -------

    """
    return 1.0 / (z + 10.0) + \
           1.0 / (z + 5.0) + \
           1.0 / (z + 2.0) + \
           1.0 / (z - 5.0)


def test_integration(Ef, tempr, cutoff=70, gf=test_gf):
    """

    Parameters
    ----------
    Ef :
        
    tempr :
        
    cutoff :
         (Default value = 70)
    gf :
         (Default value = test_gf)

    Returns
    -------

    """

    R = 1.0e10

    a1, b1 = poles_and_residues(cutoff=cutoff)
    zero_moment = 1j * R * gf(1j * R)

    ans = 0
    # Ef = -5

    betha = 1.0 / (8.617333262145e-5 * tempr)

    for j in range(len(a1)):
        if np.real(a1[j]) > 0:
            aaa = Ef + 1j / a1[j] / betha
            ans = ans + 4 * 1j / betha * gf(aaa) * b1[j]
            # print(np.imag(ans), a1[j], b1[j])

    return np.real(zero_moment * 0 + np.imag(ans))


def bf_integration(Ef, tempr, gf=test_gf):
    """

    Parameters
    ----------
    Ef :
        
    tempr :
        
    gf :
         (Default value = test_gf)

    Returns
    -------

    """

    # temp = 100
    R = 2e2
    # Ef = 2.1

    energy = np.linspace(-R+Ef, R+Ef, 3e7)
    ans = np.imag(np.trapz(test_gf(energy + 1j * 10e-5) * fd(energy, Ef, tempr), energy))

    return -2 / np.pi * ans


if __name__=='__main__':

    import matplotlib.pyplot as plt
    # ans = bf_integration(gf=test_gf)

    Ef = np.linspace(-50, 50, 150)
    ans = []
    ans1 = []
    for ef in Ef:
        ans1.append(test_integration(ef, 1, cutoff=100, gf=test_gf1))
        # ans.append(bf_integration(ef, 10, gf=test_gf))

    # plt.plot(Ef, np.array(ans))
    plt.plot(Ef, np.array(ans1), 'o-')
    plt.show()
    print(ans, ans1)

    # a1, b1 = poles_and_residues(cutoff=2)
    # a2, b2 = poles_and_residues(cutoff=10)
    # a3, b3 = poles_and_residues(cutoff=30)
    # a4, b4 = poles_and_residues(cutoff=50)
    # a5, b5 = poles_and_residues(cutoff=100)
    #
    # print(a5, b5)

    # energy = np.linspace(-5.7, 5.7, 3000)
    #
    # temp = 300
    # fd0 = fd(energy, 0, temp)
    #
    # kb = 8.61733e-5  # Boltzmann constant in eV
    # energy = energy / (kb * temp)
    #
    # ans1 = approximant(energy, a1, b1)
    # ans2 = approximant(energy, a2, b2)
    # ans3 = approximant(energy, a3, b3)
    # ans4 = approximant(energy, a4, b4)
    # ans5 = approximant(energy, a5, b5)
    #
    # plt.plot(energy, fd0)
    # plt.plot(energy, ans1)
    # plt.plot(energy, ans2)
    # plt.plot(energy, ans3)
    # plt.plot(energy, ans4)
    # plt.plot(energy, ans5)
    # plt.show()
    #
