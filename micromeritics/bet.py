"""This module provides the BET calculations on isotherm data.
See: http://micro.edu/calculations/bet.html for details.
"""

import numpy as np
import math

from . import constants as const, util

def Isotherm2BET(Prel, Qads) :
    """Calculate the BET Transform

    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)

    Returns: BET transform of the data: 1/(Qads*(1/Prel-1))"""

    return 1/(Qads*(1/Prel-1))

def Isotherm2RoquerolBET(Prel, Qads) :
     """Calculate the Rouquerol BET Transform  Qads (1-Prel)

    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)

    Returns: BET Rouquerol transform of the data"""

    return Qads*(1-Prel)

def BETIsotherm(Prel, Qads, C, Qm, csa):
    """ Return the BET Model isotherm
    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)
    csa:  Molecular cross sectional area.  (nm^2)
    Qm: Monolayer quantity adsorbed (cm^3/g STP)
    C: BET constant"""

    # NOTE: The BET equation has a pole for 0.0 <= p/p0 <= 1.0 if C is too small or negative.
    return Qm * C * Prel / ( (1.0 - Prel)*( (C - 1.0)* Prel + 1))

def bet(Prel, Qads, Pmin, Pmax, csa):
    """Run the BET surface area calulation.

    Arguments:
    Qads: Quauntity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)
    Pmin: Minimum relative pressure to use in the BET area calculation
    Pmax: Maximum relative pressure to use in the BET area calculation
    csa:  Molecular cross sectional area.  (nm^2)

    Returns a namedtouple with the following fields:
    transform:  BET transform of the data: 1/(Qads*(1/Prel-1))
    C:          BET C value
    sa:         BET Surface area (m^2/g)
    sa_err:     Uncertainty in the BET surface area.
    q_m:        Monolayer capacity (cm^3/g STP)
    line_fit:   The line fit statistics from transform vs. Prel.
    """

    Prel_fit, Qads_fit = util.restrict_isotherm(Prel, Qads, Pmin, Pmax)

    transform_all = Isotherm2BET(Prel, Qads)
    transform_fit = Isotherm2BET(Prel_fit, Qads_fit)
    lf = util.linefit(Prel_fit, transform_fit)

    C = (lf.slope + lf.y_intercept)/lf.y_intercept
    q_m = 1/(lf.slope + lf.y_intercept)
    sa = const.AVOGADRO*csa/(const.VOLGASTP*const.NM2_M2*(lf.slope + lf.y_intercept))
    sa_err = sa*(math.sqrt((lf.slope_err**2)+(lf.y_intercept_err**2))/(lf.slope+lf.y_intercept))

    Qads_model = BETIsotherm( Prel, Qads, C, q_m, csa )

    return util.make_tuple(
        "BETResults",
        Prel_all = Prel,
        Qads_all = Qads,
        Qads_model = Qads_model,
        Pmin = Pmin,
        Pmax = Pmax,
        transform_all = transform_all,
        Prel_fit = Prel_fit,
        transform_fit = transform_fit,
        C = C,
        q_m = q_m,
        sa = sa,
        sa_err = sa_err,
        line_fit=lf,
    )

def CalcBETArea(Prel, Qads, Pmin, Pmax, csa):
    """Calculates the BET Surface Area.

    Arguments:
    Qads: Quantity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)
    csa:  Molecular cross sectional area.  (nm^2)

    Returns: BET Surface area (m^2/g)
    """

    return bet(Prel, Qads, Pmin, Pmax, csa).sa
