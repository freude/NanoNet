from setuptools import setup


setup(name='nanonet',
      version='1.0',
      description='Python framework for tight-binding computations',
      author='M. V. Klymenko, J. A. Vaitkus, J. S. Smith, J. H. Cole',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['tb', 'negf', 'verbosity'],
      entry_points={
        'console_scripts': ['tb = tb.tb_script:main', 'tbmpi = tb.tbmpi_script:main', 'gf = tb.gf_script:main'],
      },
      zip_safe=False
      )
