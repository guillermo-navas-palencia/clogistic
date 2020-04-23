=========
clogistic
=========

Logistic regression with bound and linear constraints. L1, L2, SOS and Elastic-Net regularization.


This is a Python implementation of the constrained logistic regression with a scikit-learn like API. This library uses `CVXPY <https://github.com/cvxgrp/cvxpy>`_ and scipy optimizer `L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_. Currently, only binary classification is supported.

Installation
============

To install the current release of clogistic from PyPI:

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

clogistic can replace the scikit-learn LogisticRegression import:

..code-bloc:: python
   
   # from sklearn.linear_models import LogisticRegression
   from clogistic import LogisticRegression


L1-norm
-------

L1-norm with bounds
-------------------

L2-norm with bounds
-------------------

SOS
---

SOS with bounds
--------------------

Elastic-Net with bounds and constraints
---------------------------------------


Methods
=======

`scikit-learn.linear_models.LogisticRegression API <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

