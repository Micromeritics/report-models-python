"""
BJH Implementation

The 4 BJH models implemented here are:
Faas, Standard, Kruk-Jaroniec-Sayari, and Dollimore-Heal

First various fuctions are defined that are used in the BJH calculations
followed by the 4 implementations
"""

import numpy as np

from . import util, constants as const

def KelvinRadius(Prel, APF, KJS=False):
    """ Kelvin Equation used for calculating the core radius

    Args:
        Prel: Relative Pressure (float or ndarray)
        APF: Adsorbate Property Factor [A] (float)
        KJS: Whether to use the Kruk-Jaroniec-Sayari correction (boolean)

    Returns:
        KelvinR: Kelvin radius [A] (float or ndarray)
    """

    with np.errstate(divide='ignore'):
        KelvinR = -APF / np.log(Prel)

    if KJS:
        KelvinR += 3.0

    return KelvinR

def InvKelvinRadius(KelvinR, APF, KJS=False):
    """ Inverse Kelvin Equation used for calculating relative pressure

    Args:
        KelvinR: Kelvin radius [A] (float)
        APF: Adsorbate Property Factor [A] (float)
        KJS: Whether to use the Kruk-Jaroniec-Sayari correction (boolean)

    Returns:
        Prel: Relative Pressure (float)
    """

    if KJS:
        KelvinR -= 3.0

    Prel = np.exp(-APF / KelvinR)

    return Prel

def XSectArea(r):
    """ Cross Sectional Area

    Args:
        r: Radius (float)

    Returns:
        Cross Sectional Area (float)
    """

    return np.pi * (r ** 2)

def AnnulusXSectArea(innerR, outerR):
    """ Annulus Cross Sectional Area

    Args:
        innerR: Inner Radius (float)
        outerR: Outer Radius (float)

    Returns:
        Cross Sectional Area of the annulus (float)
    """

    totalA = np.pi * ((innerR + outerR) ** 2)
    innerA = np.pi * (innerR ** 2)

    return totalA - innerA

def WgtAvg(r1, r2=None):
    """ Performs Weighted Average

    Args:
        r1: ndarray of numbers e.g. np.array([1,2,3,4,5])
        r2: Optional, adds ability to do WgtAvg(4,6)

    Returns:
        WgtAvg_r: ndarray (or float) of the weighted averages
    """

    if r2 is None:
        r = r1

        r1 = r[:-1]
        r2 = r[1:]

    WgtAvg_r = r1 * r2 * (r1 + r2) / (r1 ** 2 + r2 ** 2)

    return WgtAvg_r

def ArithAvg(r1, r2=None):
    """ Performs Arithmetic Average

    Args:
        r1: ndarray of numbers e.g. np.array([1,2,3,4,5])
        r2: Optional, adds ability to do ArithAvg(4,6)

    Returns:
        ArithAvg_r: ndarray (or float) of the arithmetic averages
    """

    if r2 is None:
        r = r1

        r1 = r[:-1]
        r2 = r[1:]

    ArithAvg_r = 0.5 * (r1 + r2)

    return ArithAvg_r

def IncCalc(PoreLen, AvgPoreRad):
    """ Performs Incremental Surface Area and Volume Calculations

    Args:
        PoreLen: Pore Lengths [cm] (ndarray)
        AvgPoreRad: Average Pore Radii [A] (ndarray)

    Returns:
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
    """

    incAFactor = np.pi * 2.0 * const.A_M * const.CM_M
    incVFactor = np.pi * const.A2_CM2

    incA = PoreLen * AvgPoreRad * incAFactor
    incV = PoreLen * (AvgPoreRad ** 2) * incVFactor

    return incA, incV

def LimitResults(PoreRad, incA, incV, LowerBound=2.0, UpperBound=2000.0):
    """ Limits the pore radii and does cumulative calculations

    Args:
        PoreRad: Pore Radii [A] (ndarray)
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
        UpperBound: Optional, Maximum Pore Radius [A] (float)
        LowerBound: Optional, Minimum Pore Radius [A] (float)

    Returns a named tuple with the following fields:
        PoreRadii: Limited Pore Radii [A] (ndarray)
        incA: Limited Incremental Pore Surface Area [m2/g] (ndarray)
        cumA: Limited Cumulative Pore Surface Area [m2/g] (ndarray)
        incV: Limited Incremental Pore Volume [cm3/g] (ndarray)
        cumV: Limited Cumulative Pore Volume [cm3/g] (ndarray)
    """

    indices = np.logical_and(PoreRad >= LowerBound, PoreRad <= UpperBound)

    PoreRad = PoreRad[indices]
    incA = incA[indices]
    incV = incV[indices]

    cumA = np.cumsum(incA)
    cumV = np.cumsum(incV)

    return util.make_tuple(
        "BJHResults",
        PoreRadii=PoreRad,
        incA=incA,
        cumA=cumA,
        incV=incV,
        cumV=cumV,
    )

def faas(Prel, Qads, APF, DCF, thick_fcn, Adsorption=False):
    """ Based on MicroActive's Faas Corrected BJH

    Args:
        Prel: Relative Pressure (ndarray)
        Qads: Quantity Absorbed [cm3/g] (ndarray)
        APF: Adsorbate Property Factor [A] (float)
        DCF: Density Conversion Factor (float)
        thick_fcn: Thickness curve (function)
        Adsorption: Whether adsorption data is used (boolean)

    Returns a named tuple with the following fields:
        PoreRadii: Pore Radii [A] (ndarray)
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        cumA: Cumulative Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
        cumV: Cumulative Pore Volume [cm3/g] (ndarray)
    """

    if Adsorption:    # If adsorption data is used, reverse order
        Prel = np.flipud(Prel)
        Qads = np.flipud(Qads)

    # Number of intervals
    Invls = len(Prel) - 1

    # Number of intervals in which new pores are found
    k = 0

    # Multilayer Thickness [A]
    Thickness = thick_fcn(Prel)
    dThickness = -np.diff(Thickness)

    # Liquid Volume [cm3]
    QadsLiq = Qads * DCF
    dQadsLiq = -np.diff(QadsLiq)

    # Kelvin Radii [A]
    KelvinRad = KelvinRadius(Prel, APF)
    AvgKelvinRad = np.zeros(Invls)

    # Upper Interval Bound Pore Radii used in graph [A]
    PoreRad = np.zeros(Invls)

    # Pore Lengths [cm]
    PoreLen = np.zeros(Invls)

    for i in range(Invls):

        # Volume desorbed from thinning of previously opened pores [cm3]
        OldWall = 0.0
        for j in range(k):
            ThinningCSA = AnnulusXSectArea(AvgKelvinRad[j], dThickness[i]) # [A2]
            OldWall += ThinningCSA * const.A2_CM2 * PoreLen[j]

        PrevInvls = k

        # Capillary volume of pores opened during current interval + immediate
        # thinning of pores opened during current interval [cm3]
        CapVol = dQadsLiq[i] - OldWall

        if CapVol > 0.0:    # Desorption from cores and walls
            if i < Invls - 1:    # Not sure the reasoning for this line but it's in the c++ code

                # Current pore radius (from pore center to film wall) [A]
                PoreRad[k] = KelvinRad[i+1]

                # Average Kelvin radius [A]
                AvgKelvinRad[k] = WgtAvg(KelvinRad[i], KelvinRad[i+1])

                # Adjust AvgKelvinRad for immediate thinning during current interval [A]
                AvgKelvinRad[k] += thick_fcn(InvKelvinRadius(AvgKelvinRad[k], APF)) - Thickness[i+1]

                # Pore Length [cm]
                PoreLen[k] = CapVol / (XSectArea(AvgKelvinRad[k]) * const.A2_CM2)

                # Increment number of intervals in which new pores are found
                k += 1

        else:    # Desorption from walls only

            # Surface Area of films currently exposed [cm2]
            WallArea = 0.0
            WallAreaFactor = 2.0 * np.pi * const.A_CM

            for j in range(k):
                WallArea += PoreLen[j] * AvgKelvinRad[j] * WallAreaFactor

            # New layer thickness which wont overcompensate for volume desorbed [A]
            if WallArea != 0.0:
                dThickness[i] = (dQadsLiq[i] / WallArea) * const.CM_A

        # Adjust Radii to represent the radius from core centers to film walls
        for j in range(PrevInvls):
            AvgKelvinRad[j] += dThickness[i] # [A]
            PoreRad[j] += dThickness[i] # [A]

    # Calculate Incremental Pore Surface Area [m2] and Volume [cm3]
    incA, incV = IncCalc(PoreLen, AvgKelvinRad)

    # Limit Pore Radii and Calculate Cumulative Data
    faas_data = LimitResults(PoreRad, incA, incV)

    return faas_data

def standard(Prel, Qads, APF, DCF, thick_fcn, Adsorption, KJS=False):
    """ Based on MicroActive's Standard BJH

    Args:
        Prel: Relative Pressure (ndarray)
        Qads: Quantity Absorbed [cm3/g] (ndarray)
        APF: Adsorbate Property Factor [A] (float)
        DCF: Density Conversion Factor (float)
        thick_fcn: Thickness curve (function)
        Adsorption: Whether adsorption data is used (boolean)
        KJS: Whether to use the Kruk-Jaroniec-Sayari correction (boolean)

    Returns a named tuple with the following fields:
        PoreRadii: Pore Radii [A] (ndarray)
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        cumA: Cumulative Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
        cumV: Cumulative Pore Volume [cm3/g] (ndarray)
    """

    if Adsorption:    # If adsorption data is used, reverse order
        Prel = np.flipud(Prel)
        Qads = np.flipud(Qads)

    # Number of intervals
    Invls = len(Prel) - 1

    # Number of intervals in which new pores are found
    k = 0

    # Multilayer Thickness [A]
    Thickness = thick_fcn(Prel)

    # Liquid Volume [cm3]
    QadsLiq = Qads * DCF

    # Kelvin Radii [A]
    KelvinRad = np.zeros(Invls+1)
    KelvinRad[0] = KelvinRadius(Prel[0], APF, KJS)
    AvgKelvinRad = np.zeros(Invls)
    AvgPoreRad = np.zeros(Invls)

    # Upper Interval Bound Pore Radii (only used in graph, not calc) [A]
    PoreRad = np.zeros(Invls)

    # Pore Lengths [cm]
    PoreLen = np.zeros(Invls)

    # i0: Lower Interval Bound, i1: Upper Interval Bound
    i0 = 0
    for i1 in range(1, Invls):

        # Volume desorbed from thinning of previously opened pores [cm3]
        OldWall = 0.0

        # Change in film thickness during interval [A]
        dThickness = Thickness[i0] - Thickness[i1]

        # Change in Liquid Volume during interval (amount desorbed) [cm3]
        dQadsLiq = QadsLiq[i0] - QadsLiq[i1]

        for j in range(0, k):
            r = AvgPoreRad[j] - Thickness[i0] # [A]
            ThinningCSA = AnnulusXSectArea(r, dThickness) # [A2]

            OldWall += ThinningCSA * const.A2_CM2 * PoreLen[j] # [cm3]

        # Capillary volume of pores opened during current interval [cm3]
        CapVol = dQadsLiq - OldWall

        if (CapVol < 0.0):
            # No new pores are found, repeat loop, increment i1 but not i0
            continue

        # The following is done for intervals in which new pores are found

        # Upper bound Kelvin radius [A]
        KelvinRad[i1] = KelvinRadius(Prel[i1], APF, KJS)

        # Average Kelvin radius [A]
        AvgKelvinRad[k] = WgtAvg(KelvinRad[i0], KelvinRad[i1])

        # Upper bound pore radius [A]
        PoreRad[k] = KelvinRad[i1] + Thickness[i1]

        # Average pore radius [A]
        AvgPoreRad[k] = AvgKelvinRad[k] + thick_fcn(InvKelvinRadius(AvgKelvinRad[k], APF, KJS))

        # Pore Length [cm]
        PoreLen[k] = CapVol / (XSectArea(AvgKelvinRad[k]) * const.A2_CM2)

        # Increment number of intervals in which new pores are found
        k += 1

        # Increment lower bound
        i0 = i1

    # Calculate Incremental Pore Surface Area [m2] and Volume [cm3]
    incA, incV = IncCalc(PoreLen, AvgPoreRad)

    # Limit Pore Radii and Calculate Cumulative Data
    standard_data = LimitResults(PoreRad, incA, incV)

    return standard_data

def kjs(Prel, Qads, APF, DCF, thick_fcn, Adsorption):
    """ Based on MicroActive's Kruk-Jaroniec-Sayari corrected BJH

    Args:
        Prel: Relative Pressure (ndarray)
        Qads: Quantity Absorbed [cm3/g] (ndarray)
        APF: Adsorbate Property Factor [A] (float)
        DCF: Density Conversion Factor (float)
        thick_fcn: Thickness curve (function)
        Adsorption: Whether adsorption data is used (boolean)

    Returns a named tuple with the following fields:
        PoreRadii: Pore Radii [A] (ndarray)
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        cumA: Cumulative Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
        cumV: Cumulative Pore Volume [cm3/g] (ndarray)
    """

    # Run the standard BJH function using the Kruk-Jaroniec-Sayari correction
    kjs_data = standard(Prel, Qads, APF, DCF, thick_fcn, Adsorption, KJS=True)

    return kjs_data

def dollimore_heal(Prel, Qads, APF, DCF, thick_fcn, Adsorption):
    """ Based on J. appl. Chem., Vol. 14, pp. 109-114
    An Improved Method for the Calculation of Pore Size Distributions
    From Adsorption Data by D. Dollimore and G. R. Heal

    Args:
        Prel: Relative Pressure (ndarray)
        Qads: Quantity Absorbed [cm3/g] (ndarray)
        APF: Adsorbate Property Factor [A] (float)
        DCF: Density Conversion Factor (float)
        thick_fcn: Thickness curve (function)
        Adsorption: Whether adsorption data is used (boolean)

    Returns a named tuple with the following fields:
        PoreRadii: Pore Radii [A] (ndarray)
        incA: Incremental Pore Surface Area [m2/g] (ndarray)
        cumA: Cumulative Pore Surface Area [m2/g] (ndarray)
        incV: Incremental Pore Volume [cm3/g] (ndarray)
        cumV: Cumulative Pore Volume [cm3/g] (ndarray)
    """

    if Adsorption:  # If adsorption data is used, reverse order
        Prel = np.flipud(Prel)
        Qads = np.flipud(Qads)

    # Number of intervals
    Invls = len(Prel) - 1

    # Kelvin Radii
    KelvinRad = KelvinRadius(Prel, APF)
    AvgKelvinRad = ArithAvg(KelvinRad)

    # Multilayer Thickness
    Thickness = thick_fcn(Prel)
    dThickness = -np.diff(Thickness)
    AvgThickness = ArithAvg(Thickness)

    # Pore Radii
    PoreRad = KelvinRad + Thickness
    AvgPoreRad = AvgKelvinRad + AvgThickness

    # BJH Volume Correction Factor
    Q = (AvgPoreRad / (AvgKelvinRad + dThickness)) ** 2

    # Quantity Adsorbed
    QadsLiq = Qads * DCF
    dQadsLiq = -np.diff(QadsLiq)

    # Incremental Area and Volume
    incA = np.zeros(Invls)
    incV = np.zeros(Invls)

    # Cumulative Surface Area
    cumA = 0.0

    # Cumulative Pore Length
    cumL = 0.0

    for i in range(Invls):
        A = dThickness[i] * cumA
        B = AvgThickness[i] * dThickness[i] * 2.0 * np.pi * cumL

        # Capillary Volume
        CapVol = max(dQadsLiq - A + B)

        # Pore Volume
        incV[i] = CapVol * Q[i]

        # Pore Area
        incA[i] = 2.0 * (incV[i] / AvgPoreRad[i])

        cumA += incA[i]
        cumL += incA[i] / (np.pi * 2.0 * AvgPoreRad[i])

    dh_data = LimitResults(PoreRad[1:], incA, incV)

    return dh_data
