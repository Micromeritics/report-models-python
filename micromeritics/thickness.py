"""This module defines classes for thickness curves
See: http://micromeritics.com/Repository/Files/calculations/thickness.html for details.

The 5 thickness curves defined here can take relative pressure in the form of
single floats or ndarrays and will return a single float or ndarray, respectively.
All functions can handle relative pressures of zero which is sometimes artificially
added as a valid point

Examples:
    >>> from micromeritics import thickness as tk
    >>> halsey = tk.Halsey()
    >>> halsey(0.5)
    6.8354229927162589

    >>> halsey(np.array([0.0, 0.2, 0.4]))
    array([0.0, 5.1634236, 6.22878279])

    >>> halsey_custom = tk.Halsey(c1=3.5, c2=-5.0, c3=0.44)
    >>> halsey_custom(0.5)
    8.3493303247877826
"""

import numpy as np
from scipy import optimize

class ThicknessCurve:
    """This provides the interface to all the thickness curves. After
    construction, the object can be called like a function to evaluate
    the thickness for the relative pressure.
    """

    def __call__(self, prel):
        """ Runs when the thickness object is called

        Args:
            prel: Relative Pressure (float or ndarray)

        Returns:
            Thickness [A] (float or ndarray)
        """

        return 0.0

class KrukJaroniecSayari(ThicknessCurve):
    """An instance of a ThicknessCurve that implements the Kruk-Jaroniec-Sayari model."""
    def __init__(self, c1=60.6500, c2=0.03071, c3=0.3968):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def __call__(self, prel):
        with np.errstate(divide='ignore'):
            return (self.c1 / (self.c2 - np.log10(prel))) ** self.c3

class Halsey(ThicknessCurve):
    """An instance of a ThicknessCurve that implements the Halsey model."""
    def __init__(self, c1=3.540, c2=-5.000, c3=0.333):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def __call__(self, prel):
        with np.errstate(divide='ignore'):
            return self.c1 * (self.c2 / np.log(prel)) ** self.c3

class HarkinsJura(ThicknessCurve):
    """An instance of a ThicknessCurve that implements the Harkins-Jura model."""
    def __init__(self, c1=13.9900, c2=0.0340, c3=0.500):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def __call__(self, prel):
        with np.errstate(divide='ignore'):
            return (self.c1 / (self.c2 - np.log10(prel))) ** self.c3

class BroekhoffDeBoer(ThicknessCurve):
    """An instance of a ThicknessCurve that implements the Broekhoff-de Boer model."""
    def __init__(self, c1=-16.1100, c2=0.1682, c3=-0.1137):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def solve_thick(self, prel):
        # Note: CANNOT take an ndarray as an argument (only float)
        # This is because optimize.newton can't take an array of functions..

        # Special case for prel=0 b/c Newton can't solve g(x) = (-inf) - etc..
        if prel == 0.0:
            return 0.0

        def bdb(x):
            # Function where x: thickness, needs to be solved using Newton's method
            # optimize.newton actually uses the secant method because fprime=None
            return (np.log10(prel) - self.c2 * np.exp(self.c3 * x)) * (x ** 2) - self.c1

        return optimize.newton(bdb, x0=5.0) # Initial guess: 5.0, based on MicroActive

    def __call__(self, prel):
        if isinstance(prel, float):
            return self.solve_thick(prel)

        elif isinstance(prel, np.ndarray):
            return np.array([self.solve_thick(i) for i in prel])

class CarbonBlackSTSA:
    """An instance of a ThicknessCurve that implements the Carbon Black STSA model."""
    def __init__(self, c1=2.9800, c2=6.4500, c3=0.8800):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def __call__(self, prel):
        return self.c1 + self.c2 * prel + self.c3 * (prel ** 2)
