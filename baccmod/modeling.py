# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: modeling.py
# Purpose: Collection of function for fitting analytic model of background
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


"""
Define the background model functions for fits and associated seeds and bounds.
"""
import numpy as np

__all__ = ['bilinear_gaussian2d',
           'FIT_FUNCTION',
           'gaussian2d',
           'log_factorial',
           'log_poisson']


def log_factorial(count_map):
    """
    Returns the log of the factorial of elements of `count_map` while computing each value only once.
    Parameters
    ----------
    count_map: Array-like of integers
        Input for which we want the factorial of all elements
    Returns
    -------
        The factorial of count_map in log scale
    """
    max_input = np.max(count_map)
    all_int = np.arange(0, max_input + 1)
    all_int[0] = 1
    log_int = np.log(all_int)
    log_int_factorials = np.cumsum(log_int)
    log_factorial_count_map = log_int_factorials[count_map]
    return log_factorial_count_map


def log_poisson(x, mu, log_factorial_x):
    return -mu + x * np.log(mu) - log_factorial_x


def gaussian2d(x, y, size, x_cm, y_cm, width, length, psi):
    """
    Evaluate the bi-dimensional gaussian law.
    Parameters
    ----------
    x, y: float 1D array
        Position at which the log gaussian is evaluated
    size: float
        Integral of the 2D Gaussian at the provided coordinates
    x_cm, y_cm: float
        Center of the 2D Gaussian
    width, length: float
        Standard deviations of the 2 dimensions of the 2D Gaussian law
    psi: float
        Orientation of the 2D Gaussian
    Returns
    -------
    gauss2d: float 1D array
        Evaluation of the 2D gaussian law at (x,y)
    """
    # Compute the x and y coordinates projection in the 2D gaussian length and width coordinates
    le = (x - x_cm) * np.cos(psi) + (y - y_cm) * np.sin(psi)
    wi = -(x - x_cm) * np.sin(psi) + (y - y_cm) * np.cos(psi)
    a = 2 * length ** 2
    b = 2 * width ** 2
    # Evaluate the 2D gaussian term
    gauss2d = np.exp(-(le ** 2 / a + wi ** 2 / b))
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d


gaussian2d.default_seeds = {'x_cm': 0, 'y_cm': 0, 'width': 1, 'length': 1, 'psi': 0}
gaussian2d.default_bounds = {'x_cm': (-5, 5), 'y_cm': (-5, 5), 'width': (-5, 5), 'length': (-5, 5),
                             'psi': (0, 2 * np.pi)}


def bilinear_gaussian2d(x, y, size, x_cm, y_cm, width, length, psi, x_gradient, y_gradient):
    """
    Adds linear gradients to `gaussian2d`
    Parameters
    ----------
    x, y, size, x_cm, y_cm, width, length, psi: see `gaussian2d`
    x_gradient: float
    y_gradient: float
    """
    return (1 + x * x_gradient) * (1 + y * y_gradient) * gaussian2d(x, y, size, x_cm, y_cm, width, length, psi)


bilinear_gaussian2d.default_seeds = {'x_cm': 0, 'y_cm': 0, 'width': 1, 'length': 1, 'psi': 0, 'x_gradient': 0,
                                     'y_gradient': 0}
bilinear_gaussian2d.default_bounds = {'x_cm': (-5, 5), 'y_cm': (-5, 5), 'width': (-5, 5), 'length': (-5, 5),
                                      'psi': (0, 2 * np.pi), 'x_gradient': (-5, 5), 'y_gradient': (-5, 5)}

FIT_FUNCTION = {'gaussian2d': gaussian2d,
                'bilinear_gaussian2d': bilinear_gaussian2d}
