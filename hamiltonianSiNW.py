import tb


""" get_h_matrix computes the hamiltonian matrix for the xyz coordinate file defined
    by the variable path. """

path = 'c:\users\sammy\desktop\NanoNet\input_samples\SiNW2.xyz'


def get_h_matrix(path):
    tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    hamiltonian = tb.Hamiltonian(xyz=path, nn_distance=2.4)
    hamiltonian.initialize()
    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(PRIMITIVE_CELL)
    return hamiltonian.h_matrix


SiNW2 = get_h_matrix(path)
print (SiNW2)
