"""
==========
metrics.py
==========

Metrics from:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline


def nrmse(y_true, y_pred, norm_value):
    rmse = np.sqrt(np.sum((y_true - y_pred)**2) / len(y_true))
    return rmse / norm_value


def local_maxima(z):

    z_interp = InterpolatedUnivariateSpline(np.arange(len(z)), z, k=4)
    dz, d2z = z_interp.derivative(), z_interp.derivative(2)

    t_extrema = dz.roots()

    t_maxima = t_extrema[d2z(t_extrema) < 0]

    maxima = z_interp(t_maxima)

    return maxima
