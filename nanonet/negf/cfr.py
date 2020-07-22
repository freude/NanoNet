import numpy as np
from scipy.linalg import eig


class CFR(object):
    """ """

    val_inf = 1.0e4

    def __init__(self, cutoff=0):
        self.cutoff = cutoff
        if cutoff == 0:
            self.fd_poles_coords = []
            self.fd_poles = []
            self.num_poles = 0
        else:
            self.gen_poles_and_residues(self.cutoff)

    def fd_approximant(self, energy):
        """

        Parameters
        ----------
        energy :
            

        Returns
        -------

        """
        arg = np.array(energy)
        ans = np.zeros(arg.shape) + 0.5

        for j in range(self.num_poles):
            if self.fd_poles_coords[j] > 0:
                ans = ans - self.fd_poles[j] / (arg - 1j / self.fd_poles_coords[j]) - \
                      self.fd_poles[j] / (arg + 1j / self.fd_poles_coords[j])

        return ans

    def fd_approximant_diff(self, energy):
        """

        Parameters
        ----------
        energy :
            

        Returns
        -------

        """

        arg = np.array(energy)
        ans = np.zeros(arg.shape) + 0.5 * 0

        for j in range(self.num_poles):
            if self.fd_poles_coords[j] > 0:
                ans = ans - self.fd_poles[j] / (arg - 1j / self.fd_poles_coords[j]) ** 2 - \
                      self.fd_poles[j] / (arg + 1j / self.fd_poles_coords[j]) ** 2

        return ans

    def gen_poles_and_residues(self, cutoff=50):
        """Compute positions of poles and their residuals for the Fermi-Dirac function

        Parameters
        ----------
        cutoff :
            cutoff energy (Default value = 50)

        Returns
        -------

        """

        self.cutoff = cutoff

        b_mat = [1 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1))) for j in range(0, cutoff - 1)]
        b_mat = np.diag(b_mat, -1) + np.diag(b_mat, 1)

        poles, residues = eig(b_mat)

        residues = 0.25 * np.array([np.abs(residues[0, j]) ** 2 / (poles[j] ** 2) for j in range(residues.shape[0])])

        self.fd_poles_coords = poles
        self.fd_poles = residues
        self.num_poles = len(self.fd_poles_coords)

    def get_poles_and_residues(self):
        """ """
        return self.fd_poles_coords, self.fd_poles

    def integrate(self, gf, ef=0, tempr=300, zero_moment=-1):
        """

        Parameters
        ----------
        gf :
            param ef:
        tempr :
            param zero_moment: (Default value = 300)
        ef :
             (Default value = 0)
        zero_moment :
             (Default value = -1)

        Returns
        -------

        """

        if zero_moment == -1:
            zero_moment = 1j * CFR.val_inf * gf(1j * CFR.val_inf)

        ans = 0
        betha = 1.0 / (8.617333262145e-5 * tempr)

        for j in range(self.num_poles):
            if np.real(self.fd_poles_coords[j]) > 0:
                aaa = ef + 1j / self.fd_poles_coords[j] / betha
                ans = ans + 4 * 1j / betha * gf(aaa) * self.fd_poles[j]

        return np.real(zero_moment + np.imag(ans))

    def genetate_integration_points(self, ef, tempr):
        """

        Parameters
        ----------
        ef :
            
        tempr :
            

        Returns
        -------

        """
        ans = []
        betha = 1.0 / (8.617333262145e-5 * tempr)

        for j in range(self.num_poles):
            ans.append(ef + 1j / self.fd_poles_coords[j] / betha)

        return np.array(ans)

    def integrate1(self, gf_vals, tempr, zero_moment=0):
        """

        Parameters
        ----------
        gf_vals :
            param tempr:
        zero_moment :
            return: (Default value = 0)
        tempr :
            

        Returns
        -------

        """

        assert len(gf_vals) == len(self.fd_poles_coords)

        ans = 0
        betha = 1.0 / (8.617333262145e-5 * tempr)

        for j in range(self.num_poles):
            if np.real(self.fd_poles_coords[j]) > 0:
                ans = ans + 4 * 1j / betha * gf_vals[j] * self.fd_poles[j]

        return np.real(zero_moment + np.imag(ans))
