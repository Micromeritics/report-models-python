"""
Example file explaining the usage of /micromeritics/bjh.py
"""

import numpy as np

from micromeritics import bjh, thickness, isotherm_examples

# Sample data including relative pressure, quantity adsorbed, density conversion
# factor, and adsorbent property factor
s = isotherm_examples.silica_alumina()
prel = s.Prel
qads = s.Qads
APF = 9.53000
DCF = 0.0015468

# Because adsorption isotherm data was used
adsorption = True

# Thickness function to use
thk_func = thickness.HarkinsJura()

# Standard, KJS, and Faas bjh data
bjh_standard = bjh.standard(prel, qads, APF, DCF, thk_func, adsorption)
bjh_kjs = bjh.kjs(prel, qads, APF, DCF, thk_func, adsorption)
bjh_faas = bjh.faas(prel, qads, APF, DCF, thk_func, adsorption)

def print_volume(bjh_data, name='PythonVolume'):
    """Prints volume data that can be pasted into MicroActive's volume graph"""

    print(name + ' (A, cm3/g, cumulative)')
    for i in range(0, len(bjh_data.PoreRadii)):
        print(bjh_data.PoreRadii[i], bjh_data.cumV[i])

print_volume(bjh_standard, 'PythonStandardVolume')
