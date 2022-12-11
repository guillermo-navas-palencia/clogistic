=========
clogistic
=========

.. image::  https://travis-ci.com/guillermo-navas-palencia/clogistic.svg?branch=master
   :target: https://travis-ci.com/guillermo-navas-palencia/clogistic

.. image::  https://codecov.io/gh/guillermo-navas-palencia/clogistic/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/guillermo-navas-palencia/clogistic

.. image:: https://img.shields.io/pypi/v/clogistic?color=blue
   :target: https://img.shields.io/pypi/v/clogistic?color=blue

.. image:: https://pepy.tech/badge/clogistic
   :target: https://pepy.tech/project/clogistic

Logistic regression with bound and linear constraints. L1, L2 and Elastic-Net regularization.


This is a Python implementation of the constrained logistic regression with a scikit-learn like API. This library uses `CVXPY <https://github.com/cvxgrp/cvxpy>`_ and scipy optimizer `L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_. Currently, only **binary** classification is supported.

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
* scikit-learn>=0.20.0
* scipy


Examples
========

clogistic can flawlessly replace the scikit-learn LogisticRegression import when bounds or linear constraints are required:

.. code-block:: python
   
   # from sklearn.linear_models import LogisticRegression
   from clogistic import LogisticRegression


L1-norm / Elastic-Net
---------------------

In the unconstrained problem, the L-BFGS-B solver supports both L1 and Elastic-Net regularization.

.. code-block:: python

   >>> from clogistic import LogisticRegression
   >>> from sklearn.datasets import load_breast_cancer
   >>> X, y = load_breast_cancer(return_X_y=True)
   >>> clf = LogisticRegression(solver="L-BFGS-B", penalty="l1")
   >>> clf.fit(X, y)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=None,
                      max_iter=100, penalty='l1', solver='L-BFGS-B', tol=0.0001,
                      verbose=False, warm_start=False)
   >>> clf.predict(X[:5, :])
   array([0, 0, 0, 1, 0])
   >>> clf.predict_proba(X[:5, :])
   array([[1.00000000e+00, 1.77635684e-14],
          [9.99999984e-01, 1.61472709e-08],
          [9.99999651e-01, 3.48756416e-07],
          [1.99686878e-01, 8.00313122e-01],
          [9.99992767e-01, 7.23307080e-06]])
   >>> clf.score(X, y)
   0.9384885764499121
   >>> clf.coef_
   array([[ 0.21636945,  0.24114984,  0.60707879, -0.02554191,  0.        ,
           -0.01089683, -0.02143886, -0.00094761,  0.        ,  0.        ,
            0.        ,  0.04741039, -0.04362739, -0.08740847,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.22373333, -0.33820163, -0.30848864, -0.00795973,  0.        ,
           -0.06749937, -0.08757346, -0.01489128, -0.00660756,  0.        ]])
   >>> clf.intercept_
   array([0.02357148])

.. code-block:: python

   >>> clf = LogisticRegression(solver="L-BFGS-B", penalty="elasticnet", l1_ratio=0.5)
   >>> clf.fit(X, y)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=0.5,
                      max_iter=100, penalty='elasticnet', solver='L-BFGS-B',
                      tol=0.0001, verbose=False, warm_start=False)
   >>> clf.score(X, y)
   0.9402460456942003


L1-norm with bounds
-------------------

Add bound constraints to force all coefficients to be negative. The intercept
represents the last position of the lower and upper bound arrays ``lb``, ``ub``,
in this case, it is unconstrained.

.. code-block:: python

   >>> import numpy as np
   >>> from scipy.optimize import Bounds
   >>> lb = np.r_[np.full(X.shape[1], -1), -np.inf]
   >>> ub = np.r_[np.zeros(X.shape[1]), np.inf]
   >>> bounds = Bounds(lb, ub)
   >>> clf = LogisticRegression(solver="ecos", penalty="l1")
   >>> clf.fit(X, y, bounds=bounds)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=None,
                      max_iter=100, penalty='l1', solver='ecos', tol=0.0001,
                      verbose=False, warm_start=False)
   >>> clf.score(X, y)
   0.9507908611599297
   >>> clf.coef_
   array([[ 6.42042386e-10,  6.69614517e-10,  7.49065341e-10,
            2.47466729e-10, -7.46445480e-08, -1.66525870e-07,
           -5.07484194e-06, -9.67293096e-08, -9.94240524e-08,
           -5.10981877e-08, -6.24719977e-08, -2.53429851e-09,
           -2.07856647e-08, -5.03914527e-02, -4.44953073e-08,
           -4.26536917e-08, -4.63999149e-08, -4.53887837e-08,
           -4.58750836e-08, -4.32208857e-08, -2.25323306e-08,
           -2.32851192e-01, -1.56344127e-01,  4.11491956e-11,
           -1.82998431e-07, -9.99999982e-01, -9.99999988e-01,
           -9.99999848e-01, -9.99999947e-01, -7.78260579e-08]])
   >>> clf.intercept_
   array([25.93817947])


L2-norm with bounds
-------------------

If we choose ``penalty="l2"`` or ``penalty="none"``, the L-BFGS-B solver can handle bound constraints.

.. code-block:: python

   >>> clf = LogisticRegression(solver="L-BFGS-B", penalty="l2")
   >>> clf.fit(X, y, bounds=bounds)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=None,
                      max_iter=100, penalty='l2', solver='L-BFGS-B', tol=0.0001,
                      verbose=False, warm_start=False)
   >>> clf.score(X, y, bounds=bounds)
   0.9507908611599297
   >>> clf.coef_
   array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00, -1.25630653e-01, -4.92843035e-01,
           -5.85325868e-01, -4.06870366e-01, -1.79105954e-01,
           -4.60000473e-02, -3.22302459e-01,  0.00000000e+00,
            0.00000000e+00, -4.54736330e-02, -6.33875425e-03,
           -6.32628802e-03, -2.51268348e-02, -1.17129553e-02,
           -1.71495885e-02, -5.82817365e-04, -8.19771941e-04,
           -2.44436774e-01, -1.53861432e-01,  0.00000000e+00,
           -2.47266502e-01, -1.00000000e+00, -1.00000000e+00,
           -6.42342321e-01, -5.32446169e-01, -1.41399360e-01]])
   >>> clf.intercept_
   array([25.96760162])


Elastic-Net with bounds and constraints
---------------------------------------

If ``solver="ecos"`` or ``solver="scs"``, linear constraints are supported. First, we solve the
unconstrained problem:

.. code-block:: python

   >>> clf = LogisticRegression(solver="ecos", penalty="elasticnet", l1_ratio=0.5)
   >>> clf.fit(X, y)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=0.5,
                      max_iter=100, penalty='elasticnet', solver='ecos',
                      tol=0.0001, verbose=False, warm_start=False)
   >>> clf.coef_
   array([[ 1.09515934e+00,  1.78915210e-01, -2.88199448e-01,
            2.26253000e-02, -2.38177991e-08, -3.48595366e-08,
           -1.11789210e-01, -5.41772242e-08, -4.46703080e-08,
           -3.70030911e-09, -9.23360225e-09,  1.34197557e+00,
            2.38283098e-08, -1.02639970e-01, -2.87375705e-09,
            6.99608679e-09, -4.41159130e-09, -4.39357355e-09,
           -4.51432833e-09,  1.46276767e-09,  1.75313422e-08,
           -4.39081317e-01, -9.05714045e-02, -1.32670345e-02,
           -8.77722530e-08, -4.68697190e-01, -1.91274067e+00,
           -2.41172826e-01, -5.15782954e-01, -1.16567422e-08]])
   >>> clf.intercept_
   array([28.2732499])
   >>> clf.score(X, y)
   0.9578207381370826

Now, we require to impose bounds and a linear constraint, for example, ``-coef_[0] + coef_[1] <= 0.5``.
The constraint has the general inequality form: ``lb <= A^Tx <= ub``.

.. code-block:: python

   >>> from scipy.optimize import LinearConstraint
   >>> lb = np.array([0.0])
   >>> ub = np.array([0.5])
   >>> A = np.zeros((1, X.shape[1] + 1))
   >>> A[0, :2] = np.array([-1, 1])
   >>> A
   array([[-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.]])
   >>> constraints = LinearConstraint(A, lb, ub)
   >>> clf = LogisticRegression(solver="ecos", penalty="elasticnet", l1_ratio=0.5)
   >>> clf.fit(X, y, bounds=bounds, constraints=constraints)
   >>> clf.coef_
   array([[ 8.38950646e-10,  9.59874680e-10,  1.09096379e-09,
            3.71912590e-10, -4.85762520e-07, -2.64846257e-01,
           -8.30023820e-01, -2.06338097e-06, -3.66858725e-06,
           -1.79685666e-07, -2.68157291e-07, -3.73083163e-09,
           -3.11904337e-08, -5.04565568e-02, -1.39102635e-07,
           -1.24094215e-07, -1.43485412e-07, -1.43613114e-07,
           -1.46108738e-07, -1.31353775e-07, -6.01051773e-08,
           -2.33773767e-01, -1.54775716e-01, -5.94112471e-11,
           -3.88166017e-01, -9.99999970e-01, -9.99999980e-01,
           -9.99999695e-01, -9.99999911e-01, -5.33323276e-07]])
   >>> clf.intercept_
   array([25.95361153])
   >>> clf.score(X, y)
   0.9507908611599297
