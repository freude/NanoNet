from setuptools import setup


setup(name='tb', 
      version='0.7',
      description='Python framework for tight-binding computations',
      author='Mike Klymenko',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=['tb'],
      entry_points={
        'console_scripts': ['tb = tb.tb_script:main', 'tbmpi = tb.tbmpi_script:main', 'gf = tb.gf_script:main'],
      },
      zip_safe=False
      )
