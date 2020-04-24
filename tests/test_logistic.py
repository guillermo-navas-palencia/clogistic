import numpy as np

from pytest import approx, raises

from clogistic import LogisticRegression

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from sklearn.datasets import load_breast_cancer


def test_parameters():
    pass


def test_inputs():
    pass


def test_predict_breast_cancer():
    # Test constrained logistic regression with the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Test that all solvers with all regularizations score (>0.93) for the
    # training data
    clf_none_lbfgs = LogisticRegression(solver="lbfgs", penalty="none")
    clf_l1_lbfgs = LogisticRegression(solver="lbfgs", penalty="l1")
    clf_l2_lbfgs = LogisticRegression(solver="lbfgs", penalty="l2")
    clf_en_lbfgs = LogisticRegression(solver="lbfgs", penalty="elasticnet",
                                      l1_ratio=0.5)
    clf_none_ecos = LogisticRegression(solver="ecos", penalty="none")
    clf_l1_ecos = LogisticRegression(solver="ecos", penalty="l1")
    clf_l2_ecos = LogisticRegression(solver="ecos", penalty="l2")
    clf_en_ecos = LogisticRegression(solver="ecos", penalty="elasticnet",
                                     l1_ratio=0.5)

    for clf in (clf_none_lbfgs, clf_l1_lbfgs, clf_l2_lbfgs, clf_en_lbfgs,
                clf_none_ecos, clf_l1_ecos, clf_l2_ecos, clf_en_ecos):

        clf.fit(X, y)
        assert np.all(np.unique(y) == clf.classes_)

        pred = clf.predict(X)
        assert np.mean(pred == y) > 0.93

        probabilities = clf.predict_proba(X)
        assert probabilities.sum(axis=1) == approx(np.ones(X.shape[0]))

        pred = clf.classes_[np.argmax(clf.predict_log_proba(X), axis=1)]
        assert np.mean(pred == y) > .9


def test_predict_breast_cancer_no_intercept():
    # Test constrained logistic regression with the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Test that all solvers with all regularizations score (>0.93) for the
    # training data without intercept
    clf_l1_lbfgs = LogisticRegression(solver="lbfgs", penalty="l1",
                                      fit_intercept=False)
    clf_l2_lbfgs = LogisticRegression(solver="lbfgs", penalty="l2",
                                      fit_intercept=False)
    clf_en_lbfgs = LogisticRegression(solver="lbfgs", penalty="elasticnet",
                                      fit_intercept=False, l1_ratio=0.5)
    clf_l1_ecos = LogisticRegression(solver="ecos", penalty="l1",
                                     fit_intercept=False)
    clf_l2_ecos = LogisticRegression(solver="ecos", penalty="l2",
                                     fit_intercept=False)
    clf_en_ecos = LogisticRegression(solver="ecos", penalty="elasticnet",
                                     fit_intercept=False, l1_ratio=0.5)

    for clf in (clf_l1_lbfgs, clf_l2_lbfgs, clf_en_lbfgs, clf_l1_ecos,
                clf_l2_ecos, clf_en_ecos):

        clf.fit(X, y)
        assert np.all(np.unique(y) == clf.classes_)

        pred = clf.predict(X)
        assert np.mean(pred == y) > 0.93

        probabilities = clf.predict_proba(X)
        assert probabilities.sum(axis=1) == approx(np.ones(X.shape[0]))

        pred = clf.classes_[np.argmax(clf.predict_log_proba(X), axis=1)]
        assert np.mean(pred == y) > .9


def test_bounds():
    pass


def test_constraints():
    pass


def test_warm_start():
    pass
