from setuptools import setup
from Cython.Build import cythonize


setup(name='nanonet',
      version='0.7',
      description='Python framework for tight-binding computations',
      author='Mike Klymenko',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['tb', 'negf'],
      entry_points={
        'console_scripts': ['tb = tb.tb_script:main', 'tbmpi = tb.tbmpi_script:main', 'gf = tb.gf_script:main'],
      },
      zip_safe=False
      )
