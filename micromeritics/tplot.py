"""This module provides the t-Plot calculations on isotherm data.
See: http://micro.edu/calculations/t-Plot.html for details.
"""

import numpy as np
import math

from . import constants as const, util

def tplot(Prel, Qads, thick_fcn, tmin, tmax,
          dcf, sacf, sa):
    """Run the t-Plot Calculation

    Arguments:
    Qads: Quantity of gas adsorbed (cm^3/g STP) (numpy array)
    Prel: Relative Pressure (numpy array)
    thick_fcn: The thickness function to use (thichness in A).
    tmin: minimum thickness to use in the line fit (Angstroms)
    tmax: maximum thickness to use in the line fit (Angstroms)
    dcf: density conversion factor (unitless)
    sacf: surface area correction factor (unitless)
    sa: surface area (m^2/g)

    Returns a namedtouple with the following fields:
    Qads:       Quantity of gas adsorbed
    t:          Thickness for the relative pressures (A)
    ext_sa:     The t-Plot external surface area (m^2/g)
    ma:         The t-Plot micropore surface area (m^2/g)
    mv:         The t-Plot micropore volume (cm^3/g)

    """

    t_all = thick_fcn(Prel)

    t_fit = []
    Qads_fit = []
    for t, q in zip(t_all,Qads):
        if t >= tmin and t<= tmax:
            t_fit.append(t)
            Qads_fit.append(q)
    t_fit = np.array(t_fit)
    Qads_fit = np.array(Qads_fit)

    lf = util.linefit(t_fit, Qads_fit)

    ext_sa = (lf.slope * dcf) / sacf * 10000
    ma = sa - ext_sa
    mv = lf.y_intercept * dcf

    return util.make_tuple(
        "tPlotResults",
        t_fit = t_fit,
        Qads_fit = Qads_fit,
        t_all = t_all,
        Qads_all = Qads,
        tmin = tmin,
        tmax = tmax,
        thick_fcn = thick_fcn,
        ext_sa = ext_sa,
        tot_sa = sa,
        mv = mv,
        ma = ma,
        line_fit=lf,
    )
