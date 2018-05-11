from setuptools import setup


setup(name='tb', 
      version='0.1',
      description='Python framework for the tight-binding computations',
      author='Mike Klymenko',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['tb'],
      scripts=['tb/tb'],
      zip_safe=False,
      python_requires='=2.7.*'
      )
