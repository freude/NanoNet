# NanoNET

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/freude/NanoNet.svg)](https://travis-ci.org/freude)
[![](https://github.com/freude/NanoNet/workflows/Nanonet%20tests/badge.svg)](https://github.com/freude/NanoNet/actions?query=workflow%3A%22Nanonet+tests%22)
[![Coverage Status](https://coveralls.io/repos/github/freude/NanoNet/badge.svg?branch=master)](https://coveralls.io/github/freude/NanoNet?branch=master)
[![codecov](https://codecov.io/gh/freude/NanoNet/branch/master/graph/badge.svg)](https://codecov.io/gh/freude/NanoNet)
[![CodeFactor](https://www.codefactor.io/repository/github/freude/nanonet/badge/master)](https://www.codefactor.io/repository/github/freude/nanonet/overview/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5186e15b951d4df6b4f20c2365870b7c)](https://app.codacy.com/app/freude/NanoNet?utm_source=github.com&utm_medium=referral&utm_content=freude/NanoNet&utm_campaign=Badge_Grade_Dashboard)
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

The source distribution can be obtained from GitHub:

```bash
git clone https://github.com/freude/NanoNet.git
cd NanoNet
```

`Nanonet` requires `openmpi` to be installed in the system:

Ubuntu
 ```bash
 sudo apt-get install libopenmpi-dev
 ```
 MacOS
 ```bash
 brew install open-mpi
 ```
 All other dependencies can be installed at once by invoking the following command
 from within the source directory:

```bash
pip install -r requirements.txt
```

### Installing

In order to install the package `Nanonet` just invoke
the following line in the bash from within the source directory:

```
pip install .
```

### Running the tests

All tests may be run by invoking the command:

```
nosetests --with-doctest
```

### Examples of usage

- [Atomic chain](jupyter_notebooks/atom_chains.ipynb)
- [Huckel model](jupyter_notebooks/Hukel_model.ipynb)
- [Bulk silicon](jupyter_notebooks/bulk_silicon.ipynb)
- [Bulk silicon - initialization via an input file](jupyter_notebooks/bulk_silicon_with_input_file.ipynb)
- [Silicon nanowire](jupyter_notebooks/silicon_nanowire.ipynb)

### Python interface

Below is a short example demonstrating usage of the `tb` package.
More illustrative examples can be found in the ipython notebooks
in the directory `jupyter_notebooks` inside the source directory.

If the package is properly installed, the work starts with the import of all necessary modules:

```python
import numpy as np
import matplotlib.pyplot as plt
import nanonet.tb as tb
```

Below we demonstrate band structure computation for an infinite atomic chain with two 
atoms A and B per unit cell [--A---B--].
1. First, one needs to specify atomic species and corresponding basis sets. We assume that each atom has one s-type atomic orbital with energies -1 eV and -0.7 eV respectively. It is also possible to use predefined basis sets as
 is shown in examples in the ipython notebooks.
 
    ```python
    a = tb.Orbitals('A')
    a.add_orbital(title='s', energy=-1)
    b = tb.Orbitals('B')
    b.add_orbital(title='s', energy=-0.7)
    ```

2. Specify geometry of the system - determine position of atoms
and specify periodic boundary conditions if any. This is done by creating an object of 
the class Hamiltonian with proper arguments.
 
    ```python
    xyz_file = """2
    Atomic Chain
    A       0.0    0.0    0.0
    B       0.0    0.0    1.0
    """
    
    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    ```

2. Now one needs to specify the coupling parameters between pairs of atoms with a given diatomic symmetry and initialize the Hamiltonian - compute Hamiltonian matrix elements

    For isolated system:
        
    ```python
    tb.set_tb_params(PARAMS_A_B={'ss_sigma': 0.3})
    h.initialize()
    ```
3. Specify periodic boundary conditions:
        
    ```python
    lattice_constant = 2.0
    h.set_periodic_bc([[0, 0, lattice_constant]])
    ```
5. Specify wave vectors:
    
    ```python
    num_points = 20
    kk = np.linspace(0, np.pi/lattice_constant, num_points, endpoint=True)
    ```

6. Find the eigenvalues and eigenstates of the Hamiltonian for each wave vector.
    
    ```python
    band_sructure = []

    for jj in range(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_sructure.append(vals)
    
    band_sructure = np.array(band_sructure)
    
    ax = plt.axes()
    ax.set_title('Band structure of the atomic chain')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)', fontsize=14)
    ax.set_ylabel(r'Energy (eV)', fontsize=14)
    ax.plot(kk * lattice_constant / np.pi, np.sort(np.real(band_sructure)), 'k')
    plt.show()
    ```

7. Done. The result will appear on the screen.

<img src="https://user-images.githubusercontent.com/4588093/83520984-e86a3600-a521-11ea-920e-9f53dac680fc.png" width="350">


### Command line interface

The package is equipped with the command line tool `tb` the usage of which reads:
 
```
tb [-h] [--k_points_file K_POINTS_FILE] [--xyz XYZ] 
   [--show SHOW] [--save SAVE] 
   [--code_name CODE_NAME] param_file
    
    positional arguments:
      param_file            Path to the file in the yaml-format containing all
                            parameters needed to run computations.
    
    optional arguments:
      -h, --help            show this help message and exit
      --k_points_file K_POINTS_FILE
                            Path to the txt file containing coordinates of wave
                            vectors for the band structure computations. If not
                            specified, default values will be used.
      --xyz XYZ             Path to the file containing atomic coordinates. If
                            specified, it overrides the coordinates specified in
                            the param_files.
      --show SHOW, -S SHOW  Show figures, 0/1/2. 0 shows nothing, 1 outputs
                            figures on screen, 2 saves figures on disk without
                            showing.
      --save SAVE, -s SAVE  Save results of computations on disk, 0/1.
      --code_name CODE_NAME
                            Code name is added to the names of all saved data
                            files.
```


The results of computations will be stored in `band_structure.pkl` file in the current directory.
This file name can be modified by specifying the parameter `--code_name`.

On the computers with `mpi` functions installed, instead of `tb` one has to use its mpi-version `tbmpi`. 
The script `tbmpi` parallelises the loop running over the wave vectors.
This script can be used together with the command `mpirun` (below is an example generating 8 parallel processes):

```
mpirun -n 8 tbmpi --show=2 --save=1 --xyz=si.xyz --k_points=k_points.txt input.yaml 
```    

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


