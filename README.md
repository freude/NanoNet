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
import tb
```

Below we demostarate band structure computation for bulk silicon using empirical tight-binding method.

1. First, one needs to specify atomic species and corresponding basis sets. It is possible to use custom basis set as
 is shown in examples in the ipython notebooks. Here we use predefined basis sets.
    
    ```python
    tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S'}
    ```

2. Specify geometry of the system - determine position if atoms
and specify periodic boundary conditions if any. This is done by creating an object of 
the class Hamiltonian with proper arguments.
 
    ```python
    xyz_file = """2
    Si cell
    Si1       0.0000000000    0.0000000000    0.0000000000
    Si2       0.0000000000    0.000000000     1.3750000000
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




