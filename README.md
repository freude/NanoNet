# TB project

The project represents an extendable Python framework for 
the electronic structure computations based on 
the tight-binding method. The code can deal with both finite
and periodic system translated in one, two or three dimensions.

## Getting Started

If the package is properly installed, the work starts with the import of all necessary modules:

```python
from tb import Hamiltonian
```

Normally, computations are performed in three stages, 
each stage can be represented by one line of code:

1. Specify geometry of the system - determine position if atoms
and specify periodic boundary conditions if any. This is done by creating an object of 
the class Hamiltonian with proper arguments. 

```python
h = Hamiltonian(xyz='path_to_xyz_file', primitive_cell=[0, 0, 1])
```

2. Initialize the Hamiltonian - compute Hamiltonian matrix elements
```python
h.initialize()
```

3. Find the eigenvalues and eigenstates of the Hamiltonian
```python
eigenvalues, eigenvectors = h.diagonalize_periodic_bc([0, 0, 0])
```

Optionally data post-processing may be performed over the obtained results of computations 
that includes data visualization, computing DOS etc.
 
 

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc



