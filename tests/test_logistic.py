import numpy as np

from pytest import approx, raises

from clogistic import LogisticRegression

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from sklearn.datasets import load_breast_cancer


def test_parameters():
    # Test parameters
    X, y = load_breast_cancer(return_X_y=True)

    with raises(ValueError):
        LogisticRegression(penalty="new_penalty").fit(X, y)

    with raises(ValueError):
        LogisticRegression(penalty="elasticnet").fit(X, y)

    with raises(ValueError):
        LogisticRegression(tol=-1e-3).fit(X, y)

    with raises(TypeError):
        LogisticRegression(fit_intercept=0).fit(X, y)

    with raises(TypeError):
        LogisticRegression(class_weight=[]).fit(X, y)

    with raises(ValueError):
        LogisticRegression(class_weight="unbalanced").fit(X, y)

    with raises(ValueError):
        LogisticRegression(solver="new_solver").fit(X, y)

    with raises(ValueError):
        LogisticRegression(max_iter=-10).fit(X, y)

    with raises(TypeError):
        LogisticRegression(warm_start=1).fit(X, y)

    with raises(TypeError):
        LogisticRegression(verbose=1).fit(X, y)


def test_solver():
    X, y = load_breast_cancer(return_X_y=True)

    clfs = [LogisticRegression(solver="lbfgs", penalty="l1", warm_start=True),
            LogisticRegression(solver="lbfgs", penalty="elasticnet",
                               warm_start=True)]
    for clf in clfs:
        with raises(ValueError):
            clf.fit(X, y)

    lb = np.r_[np.full(X.shape[1], -1), -np.inf]
    ub = np.r_[np.zeros(X.shape[1]), np.inf]
    bounds = Bounds(lb, ub)

    clfs = [LogisticRegression(solver="lbfgs", penalty="l1"),
            LogisticRegression(solver="lbfgs", penalty="elasticnet")]

    for clf in clfs:
        with raises(ValueError):
            clf.fit(X, y, bounds=bounds)

    lb = np.array([0.0])
    ub = np.array([0.5])
    A = np.zeros((1, X.shape[1] + 1))
    A[0, :2] = np.array([-1, 1])
    constraints = LinearConstraint(A, lb, ub)

    for clf in clfs:
        with raises(ValueError):
            clf.fit(X, y, constraints=constraints)


def test_target():
    X, y = load_breast_cancer(return_X_y=True)

    with raises(ValueError):
        y2 = np.random.randn(y.size)
        LogisticRegression().fit(X, y2)

    with raises(ValueError):
        y2 = np.ones(y.size)
        LogisticRegression().fit(X, y2)


def test_bounds():
    X, y = load_breast_cancer(return_X_y=True)

    bounds = [(-np.inf, np.inf)] * X.shape[1]
    with raises(TypeError):
        LogisticRegression(penalty="l2").fit(X, y, bounds=bounds)

    lb = np.r_[np.full(X.shape[1], -1), -np.inf]
    ub = np.r_[np.zeros(X.shape[1]-1), np.inf]
    bounds = Bounds(lb, ub)

    with raises(ValueError):
        LogisticRegression(penalty="l2").fit(X, y, bounds=bounds)

    lb = np.r_[np.full(X.shape[1]-1, -1), -np.inf]
    ub = np.r_[np.zeros(X.shape[1]-1), np.inf]
    bounds = Bounds(lb, ub)

    with raises(ValueError):
        LogisticRegression(penalty="l2").fit(X, y, bounds=bounds)


def test_contraints():
    X, y = load_breast_cancer(return_X_y=True)

    lb = np.array([0.0])
    ub = np.array([0.5])
    A = np.zeros((1, X.shape[1] + 1))
    A[0, :2] = np.array([-1, 1])

    with raises(TypeError):
        LogisticRegression().fit(X, y, constraints=[A, lb, ub])

    lb = np.array([0.0])
    ub = np.array([0.5, 0.2])
    constraints = LinearConstraint(A, lb, ub)

    with raises(ValueError):
        LogisticRegression().fit(X, y, constraints=constraints)

    lb = np.array([0.0])
    ub = np.array([0.5])
    A = np.zeros((1, X.shape[1]))
    constraints = LinearConstraint(A, lb, ub)

    with raises(ValueError):
        LogisticRegression().fit(X, y, constraints=constraints)

    lb = np.array([0.0, 0.2])
    ub = np.array([0.5, 0.2])
    A = np.zeros((1, X.shape[1] + 1))
    constraints = LinearConstraint(A, lb, ub)

    with raises(ValueError):
        LogisticRegression().fit(X, y, constraints=constraints)


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


def test_predict_breast_cancer_bounds_constraints():
    # Test constrained logistic regression with the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    lb = np.r_[np.full(X.shape[1], -1), -np.inf]
    ub = np.r_[np.zeros(X.shape[1]), np.inf]
    bounds = Bounds(lb, ub)

    lb = np.array([0.0])
    ub = np.array([0.5])
    A = np.zeros((1, X.shape[1] + 1))
    A[0, :2] = np.array([-1, 1])
    constraints = LinearConstraint(A, lb, ub)

    # Test that all solvers with all regularizations score (>0.93) for the
    # training data
    clf_none_ecos = LogisticRegression(solver="ecos", penalty="none")
    clf_l1_ecos = LogisticRegression(solver="ecos", penalty="l1")
    clf_l2_ecos = LogisticRegression(solver="ecos", penalty="l2")
    clf_en_ecos = LogisticRegression(solver="ecos", penalty="elasticnet",
                                     l1_ratio=0.5)

    for clf in (clf_none_ecos, clf_l1_ecos, clf_l2_ecos, clf_en_ecos):

        clf.fit(X, y, bounds=bounds, constraints=constraints)
        assert np.all(np.unique(y) == clf.classes_)

        pred = clf.predict(X)
        assert np.mean(pred == y) > 0.93

        probabilities = clf.predict_proba(X)
        assert probabilities.sum(axis=1) == approx(np.ones(X.shape[0]))

        pred = clf.classes_[np.argmax(clf.predict_log_proba(X), axis=1)]
        assert np.mean(pred == y) > .9


def test_warm_start():
    X, y = load_breast_cancer(return_X_y=True)

    clf_l2_lbfgs = LogisticRegression(solver="lbfgs", penalty="l2",
                                      warm_start=True)
    clf_l2_ecos = LogisticRegression(solver="ecos", penalty="l2",
                                     warm_start=True)

    for clf in (clf_l2_lbfgs, clf_l2_ecos):
        clf.fit(X, y)
        score1 = clf.score(X, y)
        clf.fit(X, y)
        score2 = clf.score(X, y)

        assert score1 == approx(score2, rel=1e-1)


def test_class_weight():
    X, y = load_breast_cancer(return_X_y=True)

    clf_l2_lbfgs = LogisticRegression(solver="lbfgs", penalty="l2",
                                      class_weight="balanced")
    clf_l2_ecos = LogisticRegression(solver="ecos", penalty="l2",
                                     class_weight={0: 1, 1: 5})

    for clf in (clf_l2_lbfgs, clf_l2_ecos):
        clf.fit(X, y)
        pred = clf.predict(X)
        assert np.mean(pred == y) > 0.93
