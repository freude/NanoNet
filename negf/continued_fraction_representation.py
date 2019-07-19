import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from negf.hamiltonian_chain import fd


def approximant(energy, poles, residues):

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = ans - residues[j] / (arg - 1j / poles[j]) - residues[j] / (arg + 1j / poles[j])

    return ans


def approximant_diff(energy, poles, residues):

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5 * 0

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = ans - residues[j] / (arg - 1j / poles[j]) ** 2 - residues[j] / (arg + 1j / poles[j]) ** 2

    return ans


def poles_and_residues(cutoff=2):

    b_mat = [1 / (2.0 * np.sqrt((2*(j+1) - 1)*(2*(j+1) + 1))) for j in range(0, cutoff-1)]
    b_mat = np.matrix(np.diag(b_mat, -1)) + np.matrix(np.diag(b_mat, 1))

    poles, residues = eig(b_mat)

    residues = np.array(np.matrix(residues))
    # arg = np.argmax(np.abs(residues), axis=0)

    residues = 0.25 * np.array([np.abs(residues[0, j])**2 / (poles[j] ** 2) for j in range(residues.shape[0])])

    return poles, residues


if __name__=='__main__':

    a1, b1 = poles_and_residues(cutoff=2)
    a2, b2 = poles_and_residues(cutoff=10)
    a3, b3 = poles_and_residues(cutoff=30)
    a4, b4 = poles_and_residues(cutoff=50)
    a5, b5 = poles_and_residues(cutoff=100)

    energy = np.linspace(-5.7, 5.7, 3000)

    temp = 100
    fd0 = fd(energy, 0, temp)

    kb = 8.61733e-5  # Boltzmann constant in eV
    energy = energy / (kb * temp)

    ans1 = approximant(energy, a1, b1)
    ans2 = approximant(energy, a2, b2)
    ans3 = approximant(energy, a3, b3)
    ans4 = approximant(energy, a4, b4)
    ans5 = approximant(energy, a5, b5)

    plt.plot(fd0)
    plt.plot(ans1)
    plt.plot(ans2)
    plt.plot(ans3)
    plt.plot(ans4)
    plt.plot(ans5)
    plt.show()

