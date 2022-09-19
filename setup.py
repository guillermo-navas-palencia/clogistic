# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


long_description = '''
Logistic regression with bound and linear constraints. L1, L2 and
Elastic-Net regularization.
'''


# install requirements
install_requires = [
    'cvxpy>=1.0.31',
    'numpy>=1.16',
    'scipy',
    'scikit-learn>=0.20.0'
]


# Read version file
version_info = {}
with open("clogistic/_version.py") as f:
    exec(f.read(), version_info)


setup(
    name='clogistic',
    version=version_info['__version__'],
    description="Constrained Logistic Regression",
    long_description=long_description,
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    url="https://github.com/guillermo-navas-palencia/clogistic",
    packages=['clogistic'],
    platforms="any",
    include_package_data=True,
    license="Apache Licence 2.0",
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=['pytest'],
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'],
    keywords='machine-learning, logistic-regression, statistics, data-science'
)
