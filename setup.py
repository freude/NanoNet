from setuptools import setup


setup(name='tb', 
      version='0.1',
      description='Python framework for the tight-binding computations',
      author='Mike Klymenko',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['tb'],
      entry_points={
        'console_scripts': ['tb = tb.tb:main', 'tbmpi = tb.tbmpi:main', 'tbmpi = tb.gf:main'],
      },
      zip_safe=False
      )
