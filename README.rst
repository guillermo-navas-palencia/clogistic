=========
clogistic
=========

Logistic regression with bound and linear constraints. L1, L2, SOS and Elastic-Net regularization.


This is a Python implementation based on the constrained logistic regression with a scikit-learn like API. This library uses `CVXPY <https://github.com/cvxgrp/cvxpy>`_ and scipy optimizer `L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_.

Installation
============

To install the current release of clogistic:

.. code-block:: text

   pip install clogistic

To install from source, download or clone the git repository

.. code-block:: text

   git clone https://github.com/guillermo-navas-palencia/clogistic.git
   cd clogistic
   python setup.py install

Dependencies
------------

* cvxpy>=1.0.31
* numpy
* scikit-learn (>=0.20.0)
* scipy


Examples
========