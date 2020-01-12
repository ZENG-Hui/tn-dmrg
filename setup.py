#!/usr/bin/env python

from setuptools import find_packages, setup

description = ('A toy DMRG code ')

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name='tndmrg',
    version='0.1',
    url='https://github.com/orialb/tn-dmrg',
    author='Ori Alberton',
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    description=description,
    packages=find_packages(),
)
