from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(name='nano-net',
      version='1.1.9',
      description='Python framework for tight-binding computations',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='M. V. Klymenko, J. A. Vaitkus, J. S. Smith, J. H. Cole',
      author_email='mike.klymenko@rmit.edu.au',
      license='MIT',
      packages=find_namespace_packages(include=['nanonet.*']),
      entry_points={
        'console_scripts': ['tb = tb.tb_script:main', 'tbmpi = tb.tbmpi_script:main', 'gf = tb.gf_script:main'],
      },
      url = "https://github.com/freude/NanoNet",
      zip_safe=False,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires=install_requires
      )
