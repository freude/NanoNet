# NanoNET

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![](https://github.com/freude/NanoNet/workflows/Nanonet%20tests/badge.svg)](https://github.com/freude/NanoNet/actions?query=workflow%3A%22Nanonet+tests%22)
[![Coverage Status](https://coveralls.io/repos/github/freude/NanoNet/badge.svg?branch=master)](https://coveralls.io/github/freude/NanoNet?branch=master)
[![codecov](https://codecov.io/gh/freude/NanoNet/graph/badge.svg?token=A765CQUD6R)](https://codecov.io/gh/freude/NanoNet)
[![CodeFactor](https://www.codefactor.io/repository/github/freude/nanonet/badge/master)](https://www.codefactor.io/repository/github/freude/nanonet/overview/master)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/090d663fd8d6482c8c2a4c4c7358d223)](https://app.codacy.com/gh/freude/NanoNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![PyPI version](https://badge.fury.io/py/nano-net.svg)](https://badge.fury.io/py/nano-net)

<img src="https://user-images.githubusercontent.com/4588093/65398380-1f684380-ddfa-11e9-9e87-5aab6cf417b8.png" width="200">

## Introduction

The project NanoNET (Nanoscale Non-equilibrium Electron Transport) represents an extendable Python framework for 
the electronic structure computations based on 
the tight-binding method. The code can deal with both finite
and periodic systems translated in one, two or three dimensions.

All computations can be governed by means of the python application programming interface (pyAPI) or the command line interface (CLI).

## Getting Started

### Requirements

`NanoNet` requires `openmpi` to be installed in the system:

Ubuntu
 ```bash
 sudo apt-get install libopenmpi-dev
 ```
 MacOS
 ```bash
 brew install open-mpi
 ```

### Installing from PiPy

The easiest way to install `NanoNet` without tests is from the PiPy repository:

```bash
pip install nano-net
```

### Installing from sources

The source distribution can be obtained from GitHub:

```bash
git clone https://github.com/freude/NanoNet.git
cd NanoNet
```


 All other dependencies can be installed at once by invoking the following command
 from within the source directory:

```bash
pip install -r requirements.txt
```

In order to install the package `Nanonet` just invoke
the following line in the bash from within the source directory:

```
pip install .
```

### Running the tests

If the source distribution is available, all tests may be run by invoking the following command in the root directory:

```
nosetests --with-doctest
```

### Examples of usage

- [Atomic chain](https://github.com/freude/NanoNet/blob/master/jupyter_notebooks/atom_chains.ipynb)
- [Huckel model](https://github.com/freude/NanoNet/blob/master/jupyter_notebooks/Hukel_model.ipynb)
- [Bulk silicon](https://github.com/freude/NanoNet/blob/master/jupyter_notebooks/bulk_silicon.ipynb)
- [Bulk silicon - initialization via an input file](https://github.com/freude/NanoNet/blob/master/jupyter_notebooks/bulk_silicon_with_input_file.ipynb)
- [Silicon nanowire](https://github.com/freude/NanoNet/blob/master/jupyter_notebooks/silicon_nanowire.ipynb)

### Python interface

Below is a short example demonstrating usage of the `tb` package.
More illustrative examples can be found in the ipython notebooks
in the directory `jupyter_notebooks` inside the source directory.

Below we demonstrate band structure computation for a nanoribbon with four 
atoms per unit cell:

<pre>
--A--
  |
--A--
  |
--A--
  |
--A--
</pre>

0. If the package is properly installed, the work starts with the import of all necessary modules:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import nanonet.tb as tb
    from nanonet.negf.recursive_greens_functions import recursive_gf
    from nanonet.negf.greens_functions import surface_greens_function
    ```
 
1. First, one needs to specify atomic species and corresponding basis sets. We assume that each atom has one s-type atomic orbital with energy -1 eV. It is also possible to use predefined basis sets as
 is shown in examples in the ipython notebooks.
 
    ```python
    orb = tb.Orbitals('A')
    orb.add_orbital(title='s', energy=-1.0)
    ```

2. Set tight-binding parameters:
    ```python
    tb.set_tb_params(PARAMS_A_A={"ss_sigma": 1.0})
    ```

3. Define atomic coordinates for the unit cell:
    ```python
    input_file = """4
                    Nanostrip
                    A1 0.0 0.0 0.0
                    A2 0.0 1.0 0.0
                    A3 0.0 2.0 0.0
                    A4 0.0 3.0 0.0
                 """
    ```
4. Make instance of the Hamiltonian class and specify periodic boundary conditions if any:
    ```python
    h = tb.Hamiltonian(xyz=input_file, nn_distance=1.4)
    h.initialize()
    h.set_periodic_bc([[0, 0, 1.0]])
    h_l, h_0, h_r = h.get_hamiltonians()
    ``` 
  
5. Compute DOS and transmission using Green's functions:

    ```python
    energy = np.linspace(-5.0, 5.0, 150)
    dos = np.zeros((energy.shape[0]))
    tr = np.zeros((energy.shape[0]))
    
    for j, E in enumerate(energy):
        # compute surface Green's functions
        L, R = surface_greens_function(E, h_l, h_0, h_r)
        # recursive Green's functions
        g_trans, grd, grl, gru, gr_left = recursive_gf(E, [h_l], [h_0 + L + R], [h_r])
        # compute DOS
        dos[j] = np.real(np.trace(1j * (grd[0] - grd[0].conj().T)))
        # compute left-lead coupling
        gamma_l = 1j * (L - L.conj().T)
        # compute right-lead coupling
        gamma_r = 1j * (R - R.conj().T)
        # compute transmission
        tr[j] = np.real(np.trace(gamma_l @ g_trans @ gamma_r @ g_trans.conj().T)))
    ```
6. Plot DOS and transmission spectrum:
    ```python
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(energy, dos, 'k')
    ax[0].set_ylabel(r'DOS (a.u)')
    ax[0].set_xlabel(r'Energy (eV)')
    
    ax[1].plot(energy, tr, 'k')
    ax[1].set_ylabel(r'Transmission (a.u.)')
    ax[1].set_xlabel(r'Energy (eV)')
    fig.tight_layout()
    plt.show()
    ```
7. Done. The result will appear on the screen.

![gh_img](https://user-images.githubusercontent.com/4588093/88499950-c74a3100-d00a-11ea-9d0f-86fa470fa47e.png)

## Authors

- Mykhailo V. Klymenko (mike.klymenko@rmit.edu.au)
- Jackson S. Smith
- Jesse A. Vaitkus
- Jared H. Cole

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

We acknowledge support of the RMIT University, 
Australian Research Council through grant CE170100026, and
National Computational Infrastructure, which is supported by the Australian Government.

## References

[M.V. Klymenko, J.A. Vaitkus, J.S. Smith, and J.H. Cole, "NanoNET: An extendable Python framework for semi-empirical tight-binding models," *Computer Physics Communications*, Volume 259, 107676 (2021)](https://doi.org/10.1016/j.cpc.2020.107676)


