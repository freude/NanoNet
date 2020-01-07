# NanoNet project

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5186e15b951d4df6b4f20c2365870b7c)](https://app.codacy.com/app/freude/NanoNet?utm_source=github.com&utm_medium=referral&utm_content=freude/NanoNet&utm_campaign=Badge_Grade_Dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/freude/NanoNet.svg)](https://travis-ci.org/freude)
[![Coverage Status](https://coveralls.io/repos/github/freude/NanoNet/badge.svg?branch=develop)](https://coveralls.io/github/freude/NanoNet?branch=develop)
[![CodeFactor](https://www.codefactor.io/repository/github/freude/nanonet/badge/master)](https://www.codefactor.io/repository/github/freude/nanonet/overview/master)

<img src="https://user-images.githubusercontent.com/4588093/65398380-1f684380-ddfa-11e9-9e87-5aab6cf417b8.png" width="200">

## What is Nanonet?
The project represents an extendable Python framework for 
the electronic structure computations based on 
the tight-binding method. The code can deal with both finite
and periodic systems translated in one, two or three dimensions.

All computations can be governed by means of the python application programming interface (pyAPI) or the command line interface (CLI).

## Installation

### Prerequisites

The source distribution can be obtained from GitHub:

```bash
git clone https://github.com/freude/NanoNet.git
cd NanoNet
```

All dependencies can be installed at once by invoking the following command
 from within the source directory:

```bash
pip install -r requirements.txt
```

### Installing

In order to install the package `tb` just invoke
the following line in the bash from within the source directory:

```
pip install .
```

### Running the tests

All tests may be run by invoking the command:

```
nosetests --with-doctest
```

## Usage and interfaces

### Python application programming interface

Below is a short example demonstrating usage of the `tb` package.
More illustrative examples can be found in the ipython notebooks
in the directory `jupyter_notebooks` inside the source directory.

If the package is properly installed, the work starts with the import of all necessary modules:

```python
import tb
```

Below we demonstrate band structure computation for bulk silicon using empirical tight-binding method.

1. First, one needs to specify atomic species and corresponding basis sets. It is possible to use custom basis set as
 is shown in examples in the ipython notebooks. Here we use predefined basis sets.
    
    ```python
    tb.Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S'}
    ```

2. Specify geometry of the system - determine position of atoms
and specify periodic boundary conditions if any. This is done by creating an object of 
the class Hamiltonian with proper arguments.
 
    ```python
    xyz_file = """2
    Si cell
    Si1       0.0000000000    0.0000000000    0.0000000000
    Si2       1.3750000000    1.3750000000    1.3750000000
    """
    
    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=2.0)
    ```

2. Initialize the Hamiltonian - compute Hamiltonian matrix elements

    For isolated system:
        
    ```python
    h.initialize()
    ```
3. Specify periodic boundary conditions:
        
    ```python
    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0.5 * a_si, 0.5 * a_si],
                     [0.5 * a_si, 0, 0.5 * a_si],
                     [0.5 * a_si, 0.5 * a_si, 0]]
    h.set_periodic_bc(PRIMITIVE_CELL)
    ```
5. Specify wave vectors:
    
    ```python
    sym_points = ['L', 'GAMMA', 'X', 'W', 'K', 'L', 'W', 'X', 'K', 'GAMMA']
    num_points = [15, 20, 15, 10, 15, 15, 15, 15, 20]
    k = tb.get_k_coords(sym_points, num_points)
    ```

6. Find the eigenvalues and eigenstates of the Hamiltonian for each wave vector.
    
    ```python
    vals = np.zeros((sum(num_points), h.h_matrix.shape[0]), dtype=np.complex)
    
    for jj, i in enumerate(k):
        vals[jj, :], _ = h.diagonalize_periodic_bc(list(i))
   
    import matplotlib.pyplot as plt 
    plt.plot(np.sort(np.real(vals)))
    plt.show()
    ```

7. Done.

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

## License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2019 RMIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



## Acknowledgments

We acknowledge support of the RMIT University, 
Australian Research Council through grant CE170100026, and
National Computational Infrastructure, which is supported by the Australian Government.


