"""
=======
fig3.py
=======

Script for reproduction of Figure 3 of paper:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import doublescroll
from metrics import nrmse

from model import nvar, tikhonov_regression, predict

# ==========
# Parameters
# ==========

# NVAR delay
k = 2
# NVAR strides
s = 1
# Monomials order
p = 3
# Regularization parameter (from code)
alpha = 1.0e-3
# Add constant term
bias = False

# Time step duration (in time unit)
dt = 0.25
# Training time (in time unit)
train_time = 100.0
# Testing time (idem)
test_time = 800.0
# Transient time (idem): should always be > k * s
transients = 1.0

# Initial conditions of Lorenz equations (found in code)
x0 = [0.37926545, 0.058339, -0.08167691]
# Runge-Kutta method
solve_method = "RK23"
# Lyapunov time
lyap_time = 7.81

# Discretization
train_steps = round(train_time / dt)
test_steps = round(test_time / dt)
trans_steps = round(transients / dt)
lyap_steps = round(lyap_time / dt)

parameters = {
    "nvar": {"k": k, "p": p, "s": s, "alpha": alpha, "bias": bias},
    "data": {
        "attractor": "lorenz",
        "dt": dt,
        "train_time": train_time,
        "test_time": test_time,
        "transients": transients,
        "doublescroll_x0": x0,
        "doublescroll_runge_kutta": solve_method,
        "doublescroll_lyapunov_time": lyap_time,
    },
}

if __name__ == "__main__":

    # ==================
    # Dataset generation
    # ==================

    N = train_steps + test_steps + trans_steps
    X = doublescroll(N, x0=x0, h=dt, method="RK23")

    Xvar = X[:, 0].var() + X[:, 1].var() + X[:, 2].var()

    X_train = X[: train_steps + trans_steps]
    X_test = X[train_steps + trans_steps :]

    dX_train = X[1 : train_steps + trans_steps + 1] - X[: train_steps + trans_steps]
    dX_test = X[train_steps + trans_steps + 1 :] - X[train_steps + trans_steps : -1]

    dX_mean, dX_std = dX_train.mean(), dX_train.std()

    # ========
    # Training
    # ========

    lin_features, nlin_features, window = nvar(X_train, k=k, s=s, p=p)
    Wout = tikhonov_regression(
        lin_features,
        nlin_features,
        dX_train,
        alpha=alpha,
        transients=trans_steps,
        bias=bias,
    )
    target_dim = Wout.shape[0]

    # ==========
    # Evaluation
    # ==========

    # On training set
    dX_pred = predict(Wout, lin_features, nlin_features)

    print(
        "Training NRMSE:",
        nrmse(dX_train[trans_steps:], dX_pred[trans_steps:], norm_value=Xvar),
    )

    # Forecasting on testing set
    u = np.atleast_2d(X_test[0, :])
    Y = np.zeros((test_steps, target_dim))
    for i in range(test_steps):
        lin_features, nlin_features, window = nvar(u, k=k, s=s, p=p, window=window)
        u = u + predict(Wout, lin_features, nlin_features)
        Y[i, :] = u

    # ====
    # Plot
    # ====
    fig = plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(nrows=1, ncols=2)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gt.set_title("Ground truth")
    ax_gt.set_xlabel("$x$")
    ax_gt.set_ylabel("$z$")
    ax_gt.grid(False)
    ax_gt.plot(X_test[:, 0], X_test[:, 2])

    ax_pr = fig.add_subplot(gs[0, 1])
    ax_pr.set_title("NG-RC prediction")
    ax_pr.set_xlabel("$x$")
    ax_pr.set_ylabel("$z$")
    ax_pr.grid(False)
    ax_pr.plot(Y[:, 0], Y[:, 2])

    fig.savefig("results/fig3.pdf")
