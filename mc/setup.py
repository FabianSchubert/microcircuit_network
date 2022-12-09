#! /usr/bin/env python3

from distutils.core import setup

setup(name="mc",
      version="0.1",
      description="GeNN implementation of the dendritic microcircuit network",
      author="Fabian Schubert",
      packages=["mc"],
      install_requires=[
            'pygenn',
            'numpy',
            'tqdm',
            'matplotlib'
      ])
