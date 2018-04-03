# TB project

The project represents an extendable Python framework for 
the electronic structure computations based on 
the tight-binding method. The code can deal with both finite
and periodic system translated in one, two or three dimensions.

## Getting Started

### Prerequisites

The source distribution archive has to be unpacked first:

```bash
tar -xvzf tb-0.1.tar.gz
cd tb-0.1
```

All dependencies may be installed at once by invoking the following command
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

### Python interface

Below is a short example demonstrating usage of the `tb` package.
More illustrative examples can be found in the ipython notebooks
in the directory `jupyter_notebooks` inside the source directory.

If the package is properly installed, the work starts with the import of all necessary modules:

```python
import Hamil
```

Normally, computations are performed in three stages, 
each stage can be represented by one or two lines of code:

1. Specify geometry of the system - determine position if atoms
and specify periodic boundary conditions if any. This is done by creating an object of 
the class Hamiltonian with proper arguments. 
    ```python
   h = Hamiltonian(xyz='path_to_xyz_file')
    ```

2. Initialize the Hamiltonian - compute Hamiltonian matrix elements

    For isolated system:
    
    ```python
   h.initialize()
    ```
    
    In order to specify periodic boundary conditions add the lines:
    
    ```python
   a_si = 5.50
   PRIMITIVE_CELL = [[0, 0.5 * a_si, 0.5 * a_si],
                     [0.5 * a_si, 0, 0.5 * a_si],
                     [0.5 * a_si, 0.5 * a_si, 0]]
   h.set_periodic_bc(PRIMITIVE_CELL)
    ```

3. Find the eigenvalues and eigenstates of the Hamiltonian

    For the isolated system:
    ```python
   eigenvalues, eigenvectors = h.diagonalize()
    ```
    
    For the system with periodic boundary conditions:
    ```python
   wave_vector = [0, 0, 0]
   eigenvalues, eigenvectors = h.diagonalize_periodic_bc(wave_vector)
    ```

Optionally data post-processing may be performed over the obtained results of computations 
that includes data visualization, computing DOS etc.


### Command line interface

The package is equipped with the command line tool `tb` the usage of which reads:
 
```tb [-h] [--k_points_file K_POINTS_FILE] param_file```
 
    Mandatory argument:
    
    param_file
        is an file in the yaml-format containing all parameters
        needed to run computations.
    
    Optional arguments and parameters:

    --k_points_file K_POINTS_FILE
        path to the txt file containing coordinates of
        wave vectors for the band structure computations. 
        If not specified, default values will be used. 
    -h
        with this parameter the information about 
        command usage will be output.

The results of computations will be stored in `band_structure.pkl` file in the current directory.

## Running the tests

## Deployment

## Contributing

## Versioning 

## Authors

## License

## Acknowledgments




