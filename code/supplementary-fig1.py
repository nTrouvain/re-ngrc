"""
=====================
supplementary-fig1.py
=====================

Script for reproduction of Supplementary Figure 1 of paper:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, ScalarFormatter

from data import lorenz
from metrics import nrmse, local_maxima
from model import nvar, tikhonov_regression, predict

# ==========
# Parameters
# ==========

# NVAR delay
k = 2
# NVAR strides
s = 1
# Monomials order
p = 2
# Regularization parameter
alpha = 2.5e-6
# Add constant term
bias = True

# Time step duration (in time unit)
dt = 0.025
# Training time (in time unit)
train_time = 10.0
# Testing time (idem)
test_time = 1000.0
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

parameters = {
    "nvar": {"k": k, "p": p, "s": s, "alpha": alpha, "bias": bias},
    "data": {
        "attractor": "lorenz",
        "dt": dt,
        "train_time": train_time,
        "test_time": test_time,
        "transients": transients,
        "lorenz_x0": x0,
        "lorenz_runge_kutta": solve_method,
        "lorenz_lyapunov_time": lyap_time,
    },
}

if __name__ == "__main__":

    # ==================
    # Dataset generation
    # ==================

    N = train_steps + test_steps + trans_steps
    X = lorenz(N, x0=x0, h=dt, method="RK23")

    total_variance = X[:, 0].var() + X[:, 1].var() + X[:, 2].var()

    X_train = X[: train_steps + trans_steps]
    X_test = X[train_steps + trans_steps :]

    dX_train = X[1 : train_steps + trans_steps + 1] - X[: train_steps + trans_steps]

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

    # Forecasting on testing set
    u = np.atleast_2d(X_test[0, :])
    Y = np.zeros((test_steps, target_dim))
    for i in range(test_steps):
        lin_features, nlin_features, window = nvar(u, k=k, s=s, p=p, window=window)
        u = u + predict(Wout, lin_features, nlin_features)
        Y[i, :] = u

    print("Test NRMSE:", nrmse(X_test, Y, norm_value=total_variance))

    # ===============
    # Return map plot
    # ===============

    # Local interpolated maxima of z
    lorenz_maxima = local_maxima(X_test[:, 2])
    model_maxima = local_maxima(Y[:, 2])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(lorenz_maxima[:-1], lorenz_maxima[1:], label="Original Lorenz", s=3, marker="P", alpha=0.75, color="gray")
    ax.scatter(model_maxima[:-1], model_maxima[1:], label="NVAR reconstruction", s=3, marker="P", alpha=0.25, color="orangered")
    plt.legend(markerscale=3, frameon=False, fontsize=10)

    ax.spines["top"].set_color("None")
    ax.spines["right"].set_color("None")

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_formatter(ScalarFormatter())

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(ScalarFormatter())

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    ax.set_xlim(30, 47)

    ax.set_xlabel("$\mathrm{max}_{local}~~z_t$")
    ax.set_ylabel("$\mathrm{max}_{local}~~z_{t+1}$")

    # Inset axes zoom

    axins = ax.inset_axes([0.1, 0.69, 0.3, 0.3])
    axins.scatter(lorenz_maxima[:-1], lorenz_maxima[1:], s=3, marker="P", alpha=0.75, color="gray")
    axins.scatter(model_maxima[:-1], model_maxima[1:], s=3, marker="P", alpha=0.25, color="orangered")

    # sub region from original code
    x1, x2, y1, y2 = 34.6, 35.5, 35.7, 36.6
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.xaxis.set_major_locator(MultipleLocator(0.4))
    axins.yaxis.set_major_locator(MultipleLocator(0.4))

    axins.tick_params(axis='both', which='major', labelsize=6)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()

    fig.savefig("results/supplementary-fig1.pdf", bbox_inches="tight")
