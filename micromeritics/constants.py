"""This module defines physical constants that are used for the report models.

In most cases this is a local wrapper for the scipy physical constants

VOLGASTP: Volume of one mole of gas at STP (cm^3)
AVOGADRO: Avogadro's constant. Number of molecules in a mole.
Unit1_Unit2: Conversion factor from Unit1 to Unit2
    e.g. CM2_A2 -> square centimeters to square angstroms
"""

import scipy.constants

VOLGASTP  = pow(10,6) * scipy.constants.physical_constants['molar volume of ideal gas (273.15 K, 101.325 kPa)'][0]
AVOGADRO  = scipy.constants.N_A

NM2_M2    = 1.0e18
CM_M      = 1.0e-2
M_CM      = 1.0e2
A_M       = 1.0e-10
M_A       = 1.0e10
A_CM      = 1.0e-8
CM_A      = 1.0e8
A2_CM2    = 1.0e-16
CM2_A2    = 1.0e16
