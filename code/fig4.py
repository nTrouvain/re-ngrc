"""
=======
fig4.py
=======

Script for reproduction of Figure 4 of paper:

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

from data import lorenz
from metrics import nrmse

from model import nvar, tikhonov_regression, predict

# ==========
# Parameters
# ==========

# NVAR delay
k = 4
# NVAR strides
s = 5
# Monomials order
p = 2
# Regularization parameter
alpha = 0.05
# Add constant term
bias = True

# Time step duration (in time unit)
dt = 0.05
# Training time (in time unit)
train_time = 20.0
# Testing time (idem)
test_time = 45.0
# Transient time (idem): should always be > k * s
transients = 5.0

# Initial conditions of Lorenz equations (found in code)
x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]
# Runge-Kutta method
solve_method = "RK23"
# Lyapunov time
lyap_time = 1.1  # 1.104 in code

# Discretization
train_steps = round(train_time / dt)
test_steps = round(test_time / dt)
trans_steps = round(transients / dt)
lyap_steps = round(lyap_time / dt)

if __name__ == "__main__":

    # ==================
    # Dataset generation
    # ==================

    N = train_steps + test_steps + trans_steps
    X = lorenz(N, x0=x0, h=dt, method=solve_method)

    Zvar = X[:, 2].var()  # only z component will be evaluated

    XY_train = X[: train_steps + trans_steps, :2]
    XY_test = X[train_steps + trans_steps :, :2]

    Z_train = X[: train_steps + trans_steps, 2:]
    Z_test = X[train_steps + trans_steps :, 2:]

    # ========
    # Training
    # ========

    lin_features, nlin_features, window = nvar(XY_train, k=k, s=s, p=p)
    Wout = tikhonov_regression(
        lin_features,
        nlin_features,
        Z_train,
        alpha=alpha,
        transients=trans_steps,
        bias=bias,
    )

    # ==========
    # Evaluation
    # ==========

    # Run on training set
    Z_pred = predict(Wout, lin_features, nlin_features)

    print(
        "Training NRMSE:",
        nrmse(Z_train[trans_steps:], Z_pred[trans_steps:], norm_value=Zvar),
    )

    # Run on testing set
    lin_features, nlin_features, window = nvar(XY_test, k=k, s=s, p=p, window=window)
    Z_pred_test = predict(Wout, lin_features, nlin_features)

    # ====
    # Plot
    # ====
    total_time = train_time + test_time
    timepoints = np.linspace(train_time, total_time, test_steps)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    ax.plot(timepoints, Z_test, label="True value", color="gray")
    ax.plot(timepoints, Z_pred_test, label="Inferered value", color="orangered", alpha=0.7)

    ax.spines["top"].set_color("None")
    ax.spines["right"].set_color("None")

    ax.set_ylabel("$z_t$")

    axins = ax.inset_axes([0.0, -0.35, 1.0, 0.25], sharex=ax)
    axins.plot(timepoints, np.abs(Z_test - Z_pred_test), label="Absolute deviation", color="gray")

    axins.set_ylabel("$|z_t - \hat{z}_t|$")
    axins.set_xlabel("Timestep")

    axins.spines["top"].set_color("None")
    axins.spines["right"].set_color("None")

    fig.legend(frameon=False)

    plt.tight_layout()

    fig.savefig("results/fig4.pdf", bbox_inches="tight")
