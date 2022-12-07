"""
=======
data.py
=======

Timeseries generation functions for Lorenz and double-scroll attractors, as used in:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

These functions are based on `reservoirpy.datasets` module (v0.3.5).

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""
from typing import Union

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(
    n_timesteps: int,
    x0: Union[np.ndarray, list],
    h: float,
    rho: float = 28.0,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    **kwargs,
) -> np.ndarray:
    """Lorenz attractor timeseries as defined by Lorenz in 1963.

    .. math::
        \\frac{\\mathrm{d}x}{\\mathrm{d}t} &= \\sigma (y-x) \\\\
        \\frac{\\mathrm{d}y}{\\mathrm{d}t} &= x(\\rho - z) - y \\\\
        \\frac{\\mathrm{d}z}{\\mathrm{d}t} &= xy - \\beta z

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to generate.
    rho : float, default to 28.0
        :math:`\\rho` parameter of the system.
    sigma : float, default to 10.0
        :math:`\\sigma` parameter of the system.
    beta : float, default to 8/3
        :math:`\\beta` parameter of the system.
    x0 : array-like of shape (3,), default to [1.0, 1.0, 1.0]
        Initial conditions of the system.
    h : float, default to 0.03
        Time delta between two discrete timesteps.
    **kwargs:
        Other parameters to pass to the `scipy.integrate.solve_ivp`
        solver.

    Returns
    -------
    array of shape (n_timesteps, 3)
        Lorenz attractor timeseries.

    References
    ----------
    E. N. Lorenz, ‘Deterministic Nonperiodic Flow’,
    Journal of the Atmospheric Sciences, vol. 20, no. 2,
    pp. 130–141, Mar. 1963,
    doi: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.
    """

    def lorenz_diff(t, state):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    t_eval = np.arange(0.0, n_timesteps * h, h)

    sol = solve_ivp(
        lorenz_diff, y0=x0, t_span=(0.0, n_timesteps * h), t_eval=t_eval, **kwargs
    )

    return sol.y.T

def doublescroll(
    n_timesteps: int,
    r1: float = 1.2,
    r2: float = 3.44,
    r4: float = 0.193,
    ir: float = 2 * 2.25e-5,
    beta: float = 11.6,
    x0: Union[list, np.ndarray] = [0.37926545, 0.058339, -0.08167691],
    h: float = 0.25,
    **kwargs,
) -> np.ndarray:
    """Double scroll attractor timeseries,
    a particular case of multiscroll attractor timeseries.

    .. math::
        \\frac{\\mathrm{d}V_1}{\\mathrm{d}t} &= \\frac{V_1}{R_1} - \\frac{\\Delta V}{R_2} -
        2I_r \\sinh(\\beta\\Delta V) \\\\
        \\frac{\\mathrm{d}V_2}{\\mathrm{d}t} &= \\frac{\\Delta V}{R_2} +2I_r \\sinh(\\beta\\Delta V) - I\\\\
        \\frac{\\mathrm{d}I}{\\mathrm{d}t} &= V_2 - R_4 I
    where :math:`\\Delta V = V_1 - V_2`.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to generate.
    r1 : float, default to 1.2
        :math:`R_1` parameter of the system.
    r2 : float, default to 3.44
        :math:`R_2` parameter of the system.
    r4 : float, default to 0.193
        :math:`R_4` parameter of the system.
    ir : float, default to 2*2e.25e-5
        :math:`I_r` parameter of the system.
    beta : float, default to 11.6
        :math:`\\beta` parameter of the system.
    x0 : array-like of shape (3,), default to [0.37926545, 0.058339, -0.08167691]
        Initial conditions of the system.
    h : float, default to 0.01
        Time delta between two discrete timesteps.

    Returns
    -------
    array of shape (n_timesteps, 3)
        Multiscroll attractor timeseries.

    References
    ----------
    G. Chen and T. Ueta, ‘Yet another chaotic attractor’,
    Int. J. Bifurcation Chaos, vol. 09, no. 07, pp. 1465–1466,
    Jul. 1999, doi: 10.1142/S0218127499001024.
    """

    def doublescroll(t, state):
        V1, V2, i = state

        dV = V1 - V2
        factor = (dV / r2) + ir * np.sinh(beta * dV)
        dV1 = (V1 / r1) - factor
        dV2 = factor - i
        dI = V2 - r4 * i

        return dV1, dV2, dI

    t_eval = np.arange(0.0, n_timesteps * h, h)

    sol = solve_ivp(
        doublescroll, y0=x0, t_span=(0.0, n_timesteps * h), t_eval=t_eval, **kwargs
    )

    return sol.y.T
