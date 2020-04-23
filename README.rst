=========
clogistic
=========

Logistic regression with bound and linear constraints. L1, L2 and Elastic-Net regularization.


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

clogistic can flawlessly replace the scikit-learn LogisticRegression import when constraints are required:

.. code-block:: python
   
   # from sklearn.linear_models import LogisticRegression
   from clogistic import LogisticRegression


L1-norm
-------

.. code-block:: python

   >>> from clogistic import LogisticRegression
   >>> from sklearn.datasets import load_breast_cancer
   >>> X, y = load_breast_cancer(return_X_y=True)
   >>> clf = LogisticRegression(solver="lbfgs", penalty="l1")
   >>> clf.fit(X, y)
   LogisticRegression(C=1.0, class_weight=None, fit_intercept=True, l1_ratio=None,
                      max_iter=100, penalty='l1', solver='lbfgs', tol=0.0001,
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


L1-norm with bounds
-------------------

L2-norm with bounds
-------------------


Elastic-Net with bounds and constraints
---------------------------------------


Methods
=======

`scikit-learn.linear_models.LogisticRegression API <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

