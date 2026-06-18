import numpy as np
import numpy.typing as npt
from nanonet.tb.aux_functions import fd
from nanonet.tb.constants import kB_eV


class Transport(object):

    def __init__(self, **kwargs):
        self.tempr = kwargs.get('tempr', 300)

    def current(self, energy: np.ndarray, T: np.ndarray, Ef_L: float | int, Ef_R: float | int) -> float | int:
        de = energy[1] - energy[0]
        current = np.trapezoid(T[None, :, :] * (fd(energy, Ef_L[...,None], self.tempr) - fd(energy, Ef_R[...,None], self.tempr))[:,None,:], dx=de, axis=2)
        return np.squeeze(current)

    def current_linearized(self, energy: np.ndarray, T: np.ndarray, Ef_L: float | int, Ef_R: float | int):

        de = energy[1] - energy[0]
        current = np.trapezoid(T * self.linearized_diff_fd(energy, 0.5 * (Ef_L + Ef_R), self.tempr), dx=de)
        return current

    @staticmethod
    def linearized_diff_fd(energy, ef_ave, tempr):

        return -(1.0 / (4 * kB_eV * tempr)) / ((np.cosh((energy - ef_ave) / (2 * kB_eV * tempr)))**2)


class Transport2D_FET(Transport):

    def __init__(self, **kwargs):
        super(Transport2D_FET, self).__init__(**kwargs)
        self.band_gap = kwargs.get('band_gap', 0)
        self.ef_left = kwargs.get('ef_left', 0)
        self.ef_right = kwargs.get('ef_right', 0)
        self.energy_window = kwargs.get('ef_right', 0.25)

    def transmission(self, energy):
        return np.heaviside(energy - self.band_gap, 0.5)

    def current(self, voltage_sd, voltage_g):

        voltage_sd = np.atleast_1d(voltage_sd)
        voltage_g = np.atleast_1d(voltage_g)

        ef_zero = 0.5 * (self.ef_left + self.ef_right)
        energy = np.linspace(ef_zero - self.energy_window, ef_zero + self.energy_window, 100)

        return super().current(energy,
                                  self.transmission(energy - voltage_g[..., None]),
                                  self.ef_left + 0.5 * voltage_sd,
                                  self.ef_right - 0.5 * voltage_sd)

    def current_linearized(self, voltage_sd, voltage_g):

        ef_zero = 0.5 * (self.ef_left + self.ef_right)
        energy = np.linspace(ef_zero - self.energy_window, ef_zero + self.energy_window, 100)

        current = super().current(energy,
                                  self.transmission(energy - voltage_g),
                                  self.ef_left + 0.5 * voltage_sd,
                                  self.ef_right - 0.5 * voltage_sd)

        if current.ndim == 0:
            return current.item()

        return current


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # I = Transport2D_FET().current(0.01, 0)
    # print(I)

    print("----------------------------------------")

    v_sd = 0.01
    v_g = -np.linspace(-0.3, 0.3, 50)
    I = Transport2D_FET().current(v_sd, v_g)
    plt.plot(I)
    plt.show()

    print("----------------------------------------")

    v_sd = np.linspace(-0.01, 0.01, 50)
    v_g = 0
    I = Transport2D_FET().current(v_sd, v_g)
    plt.plot(I)
    plt.show()

    print("----------------------------------------")

    v_sd = np.linspace(-0.01, 0.01, 30)
    v_g = np.linspace(-0.1, 0.1, 50)
    I = Transport2D_FET().current(v_sd, v_g)
    plt.contourf(I, 100)
    plt.show()