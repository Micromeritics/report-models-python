"""This module provides the Langmuir calculations on isotherm data.
See: http://micro.edu/calculations/langmuir.html for details.
"""

import numpy as np
import math

from . import constants as const, util

def LangmuirIsotherm(Pabs, Qads, b, Qm):
    """ Return the Langmuir Model isotherm
    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Pabs: Absolute Pressure (numpy array)
    Qm: Monolayer quantity adsorbed (cm^3/g STP)
    b: Langmuir constant (energy of adsorption)"""

    return Qm * b * Pabs / (1.0 + b*Pabs)

def Isotherm2Langmuir(Pabs, Qads):
    """ Return the Langmuir transformed isotherm
    """
    return Pabs/Qads

def langmuir(Pabs, Qads, Pmin, Pmax, csa):
    """Run the Langmuir surface area calulation.

    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Pabs: Absolute Pressure (numpy array)
    Pmin: Minimum relative pressure to use in the Langmuir area calculation
    Pmax: Maximum absative pressure to use in the Langmuir area calculation
    csa:  Molecular Cross-sectional area (nm^2)

    Returns a namedtouple with the following fields:
    transform:  Langmuir transform of the data:
    b:          Langmuir b value
    sa:         Langmuir Surface area (m^2/g)
    sa_err:     Uncertainty in the Langmuir surface area.
    qm:        Monolayer capacity (cm^3/g STP)
    line_fit:   The line fit statistics from transform vs. Pabs.
    """

    Pabs_fit, Qads_fit = util.restrict_isotherm(Pabs, Qads, Pmin, Pmax)

    transform_all = Isotherm2Langmuir(Pabs, Qads)
    transform_fit = Isotherm2Langmuir(Pabs_fit, Qads_fit)
    lf = util.linefit(Pabs_fit, transform_fit)

    sa = csa * const.AVOGADRO / (const.VOLGASTP * const.NM2_M2 * lf.slope)
    sa_err = sa * lf.slope_err / lf.slope
    qm = 1.0 / lf.slope;
    b = 1.0 / (qm * lf.y_intercept)

    Qads_model = LangmuirIsotherm(Pabs, Qads, b, qm)
    return util.make_tuple(
        "LangmuirResults",
        Pabs_all = Pabs,
        Qads_all = Qads,
        Qads_model = Qads_model,
        Pmin = Pmin,
        Pmax = Pmax,
        transform_all = transform_all,
        Pabs_fit = Pabs_fit,
        transform_fit = transform_fit,
        b = b,
        qm = qm,
        sa = sa,
        sa_err = sa_err,
        line_fit=lf,
    )

def CalcLangmuirArea(Pabs, Qads, Pmin, Pmax, csa):
    """Convenience method for calculating the Langmuir Surface Area.

    Arguments:
    Qads: Quantity of gas adsorbed (cm^3/g STP) (numpy array)
    Pabs: Absolute Pressure (numpy array)

    Returns: Langmuir Surface area (m^2/g)
    """
    return langmuir(Pabs, Qads, Pmin, Pmax, csa).sa
