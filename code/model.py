"""
========
model.py
========

Non-linear Vector Autoregressive (NVAR) model from paper:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""
import itertools as it

import numpy as np
import scipy.linalg

from numpy.testing import assert_array_equal


def nvar(X, k, s, p, window=None):
    """Apply Non-linear Vecror Autoregressive model (NVAR) to timeseries.

    Parameters
    ----------
    X : numpy.ndarray
        Multivariate timeseries of shape (timesteps, n_dimensions)
    k : int
        Delay of the NVAR
    s : int
        Strides of the NVAR
    p : int
        Order of the non-linear features (i.e. monomials order)

    Returns
    -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray
            Linear features vector, non-linear features vector,
            last sliding window over the signal.
    """
    if k < 1:
        raise ValueError("k should be >= 1.")
    if s < 1:
        raise ValueError("s should be >= 1.")
    if p < 1:
        raise ValueError("p should be > 0.")
    if X.ndim < 2:
        X = X.reshape(-1, 1)

    # Inputs must be of shape (timesteps, dimension)
    n_steps, n_dim = X.shape

    lin_dim = n_dim * k  # Linear features dimension

    # Finding all monomials of order p in lin_features requires finding all
    # unique combinations of p lin_features elements, with replacement.
    lin_idx = np.arange(lin_dim)
    monom_idx = np.array(list(it.combinations_with_replacement(lin_idx, p)))

    nlin_dim = monom_idx.shape[0]

    # A sliding window to store all lagged inputs, including discarded ones.
    # By default, the window is initialized with zeros,
    # transient features will have unexpected zeros.
    win_dim = (k - 1) * s + 1  # lagged window dimension
    if window is None:
        window = np.zeros((win_dim, n_dim))
    else:
        if window.shape != (win_dim, n_dim):
            raise ValueError(
                f"window must be of shape ({win_dim}, {n_dim}) "
                f"but is of shape {window.shape}."
            )

    # Linear features and non-linear features vectors.
    lin_features = np.zeros((n_steps, lin_dim))
    nlin_features = np.zeros((n_steps, nlin_dim))

    for i in range(n_steps):
        window = np.roll(window, -1, axis=0)
        window[-1, :] = X[i]

        lin_feat = window[::s, :].flatten()
        nlin_feat = np.prod(lin_feat[monom_idx], axis=1)

        lin_features[i, :] = lin_feat
        nlin_features[i, :] = nlin_feat

    return lin_features, nlin_features, window


def tikhonov_regression(lin_features, nlin_features, target, alpha, transients, bias):
    """Performs Tikhonov linear regression (with L2 regularization) between
    NVAR features and a target signal to create a readout weight matrix.

    Parameters
    ----------
    lin_features : numpy.ndarray
        NVAR linear features of shape (timesteps, linear_dimension)
    nlin_features : numpy.ndarray
        NVAR non-linear features of shape (timesteps, nonlinear_dimension)
    target : numpy.ndarray
        Target signal, of shape (timesteps, target_dimension)
    alpha : float
        Regularization coefficient
    transients : int
        Number of timesteps to consider as transients (will be discarded before
        linear regression)
    bias : bool
        If True, add a constant term to NVAR features to compute intercept
        during linear regression

    Returns
    -------
        numpy.ndarray
            Readout weights matrix of shape
            (target_dimension, linear_dimension + non_linear_dimension + bias)
    """

    n_steps = len(lin_features) - transients

    tot_features = np.c_[lin_features, nlin_features][transients:]
    Y = target[transients:]

    if Y.ndim < 2:
        Y = Y.reshape(-1, 1)

    # Add a constant term c = 1 to all features
    if bias:
        c = np.ones((n_steps, 1))
        tot_features = np.c_[c, tot_features]

    target_dim = Y.shape[1]
    features_dim = tot_features.shape[1]
    Wout = np.zeros((target_dim, features_dim))

    # Wout = Y.Otot^T.(Otot.Otot^T + alphaId)^-1
    # (inverted all transpose as we prefer having time on the first axis)
    YXt = np.dot(Y.T, tot_features)
    XXt = np.dot(tot_features.T, tot_features)
    ridge = alpha * np.identity(len(XXt), dtype=np.float64)

    Wout[:] = np.dot(YXt, scipy.linalg.pinvh(XXt + ridge))

    return Wout


def predict(Wout, lin_features, nlin_features):
    """Use the NVAR features and the learned readout matrix for inference.

    Parameters
    ----------
    Wout : numpy.ndarray
        Readout matrix of shape
        (target_dimension, linear_dimension + non_linear_dimension + bias)
    lin_features : numpy.ndarray
        NVAR linear features of shape (timesteps, linear_dimension)
    nlin_features : numpy.ndarray
        NVAR non-linear features of shape (timesteps, nonlinear_dimension)

    Returns
    -------
        numpy.ndarray
            A predicited signal of shape (timesteps, target_dimensions)
    """

    tot_features = np.c_[lin_features, nlin_features]

    # If bias:
    if Wout.shape[1] == tot_features.shape[1] + 1:
        W, bias = Wout[:, 1:], Wout[:, :1]
    else:
        W, bias = Wout, np.zeros((Wout.shape[0], 1))

    # Transpose to keep time in first axis.
    return (np.dot(W, tot_features.T) + bias).T


# ==========
# Some tests
# ==========
if __name__ == "__main__":
    X = np.arange(10).reshape(-1, 1)
    lin_feat, nlin_feat, _ = nvar(X, k=3, s=3, p=3)

    assert_array_equal(
        lin_feat,
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, 4],
                [0, 2, 5],
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
            ]
        )
    )

    assert_array_equal(
        nlin_feat,
        np.array(
            [
                [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
                [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,],
                [  0,   0,   0,   0,   0,   0,   0,   0,   0,   8,],
                [  0,   0,   0,   0,   0,   0,   0,   0,   0,  27,],
                [  0,   0,   0,   0,   0,   0,   1,   4,  16,  64,],
                [  0,   0,   0,   0,   0,   0,   8,  20,  50, 125,],
                [  0,   0,   0,   0,   0,   0,  27,  54, 108, 216,],
                [  1,   4,   7,  16,  28,  49,  64, 112, 196, 343,],
                [  8,  20,  32,  50,  80, 128, 125, 200, 320, 512,],
                [ 27,  54,  81, 108, 162, 243, 216, 324, 486, 729,],
            ]
        )
    )

    Wout = tikhonov_regression(
        lin_feat, nlin_feat, X, transients=3, alpha=1e-6, bias=True
    )

    lin_dim = lin_feat.shape[1]
    nlin_dim = nlin_feat.shape[1]
    assert Wout.shape == (1, lin_dim + nlin_dim + 1)

    y = predict(Wout, lin_feat, nlin_feat)
    assert y.shape == X.shape
