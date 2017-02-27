"""
User interface for the BJH Jupyter notebook

Includes toggles for bjh correction, thickness curve, and isotherm range
"""

# Calculations
import numpy as np
import scipy.interpolate as intrp

from micromeritics import isotherm_examples as ex, thickness, bjh, util

# Graphing
import matplotlib.pyplot as plt
import matplotlib.ticker

# Display
from IPython.display import display, clear_output
from ipywidgets import HBox, Layout, HTML
import ipywidgets as widgets

s = ex.silica_alumina()
prel = s.Prel
qads = s.Qads
APF = 9.53000
DCF = 0.0015468
adsorption = True

def bjh_plot(x=None):
    """
    This was created assuming the matplotlib inline backend is used (%matplotlib inline)
    If %matplotlib notebook is used it may be more efficient to create a graph
    once and use the ipywidgets to update/change the data

    Takes an argument b/c wgt.observe calls BJHPlot(wgt.value) but we need the
    value of all the wgts to do the calculation so we use global vars instead
    """

    # Remove old graph but wait until new graph is made so the page won't change position
    clear_output(wait=True)

    fig = plt.figure(figsize=(15.5, 11))

    # Adjust space between VolPlot and AreaPlot
    fig.subplots_adjust(hspace=0.25)

# Create VolPlot
    vol_plot = fig.add_subplot(2, 1, 1)
    vol_plot.set_title('Cumulative Volume')
    vol_plot.grid()

    # Set x-axis (pore radius)
    vol_plot.set_xlabel('Pore Radius [$\AA$]')
    vol_plot.set_xscale('log')
    vol_plot.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    vol_plot.set_xticks([5, 10, 50, 100, 500, 1000])

    # Set y-axis 1 (cumulative pore volume)
    vol_plot.set_ylabel('Pore Volume [cm3/g]')

# Create AreaPlot
    area_plot = fig.add_subplot(2, 1, 2)
    area_plot.set_title('Cumulative Area')
    area_plot.grid()

    # Set x-axis (pore radius)
    area_plot.set_xlabel('Pore Radius [$\AA$]')
    area_plot.set_xscale('log')
    area_plot.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    area_plot.set_xticks([5, 10, 50, 100, 500, 1000])

    # Set y-axis 1 (cumulative pore area)
    area_plot.set_ylabel('Pore Area [m2/g]')

# Get currently selected widget values
    selected_cor = cor_wgt.value
    selected_thk = thk_wgt.value
    selected_iso0 = iso_wgt.value[0]
    selected_iso1 = iso_wgt.value[1]

# Restrict Prel if needed
    if selected_iso0 == 0.0 and selected_iso1 == 1.0:
        r_prel, r_qads = prel, qads
    else:
        r_prel, r_qads = util.restrict_isotherm(prel, qads, selected_iso0, selected_iso1)

# Run BJH Calculation
    bjh_data = bjh_dict[selected_cor](r_prel, r_qads, APF, DCF, thk_dict[selected_thk], adsorption)
    r, v, a = bjh_data.PoreRadii, bjh_data.cumV, bjh_data.cumA

# Plot Volume
    vol_plot.plot(r, v, 'or-', alpha=0.7)
    vol_plot.autoscale_view(True, True, True)

    # Set Volume y-axis 2 (derivative)
    d_vol_plot = vol_plot.twinx()
    d_vol_plot.set_ylabel('Derivative [cm3/g]')

    # Calculate Volume Derivative (this might differ from MicroActive)
    smooth_v = intrp.splrep(np.log(r[::-1]), v[::-1], s=0.0002)
    dV = intrp.splev(np.log(r[::-1]), smooth_v, der=1)
    dV = -dV[::-1]*np.log(10)

    # Plot Volume Derivative
    d_vol_plot.plot(r, dV, 'oc-', alpha=0.7)
    d_vol_plot.autoscale_view(True, True, True)

# Plot Area
    area_plot.plot(r, a, 'or-', alpha=0.7)
    area_plot.autoscale_view(True, True, True)

    # Set Area y-axis 2 (derivative)
    d_area_plot = area_plot.twinx()
    d_area_plot.set_ylabel('Derivative [m2/g]')

    # Calculate Area Derivative (this might differ from MicroActive)
    smooth_a = intrp.splrep(np.log(r[::-1]), a[::-1], s=0.0002)
    dA = intrp.splev(np.log(r[::-1]), smooth_a, der=1)
    dA = -dA[::-1]*np.log(10)

    # Plot Area Derivative
    d_area_plot.plot(r, dA, 'oc-', alpha=0.7)
    d_area_plot.autoscale_view(True, True, True)

# Widget dictionary
bjh_dict = {
    1: bjh.standard,
    2: bjh.kjs,
    3: bjh.faas
}

thk_dict = {
    1: thickness.KrukJaroniecSayari(),
    2: thickness.Halsey(),
    3: thickness.HarkinsJura(),
    4: thickness.BroekhoffDeBoer(),
    5: thickness.CarbonBlackSTSA()
}

def bjh_display():

    global cor_wgt
    global thk_wgt
    global iso_wgt

    # Widgets
    cor_wgt = widgets.ToggleButtons(
        options={'Standard': 1, 'Kruk-Jaroniec-Sayari': 2, 'Faas': 3},
        value=1
    )
    cor_wgt.observe(bjh_plot, names='value')

    thk_wgt = widgets.ToggleButtons(
        options={'Kruk-Jaroniec-Sayari': 1, 'Halsey': 2, 'Harkins and Jura': 3,
                 'Broekhoff-de Boer': 4, 'Carbon Black STSA': 5},
        value=3
    )
    thk_wgt.observe(bjh_plot, names='value')

    iso_wgt = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0, max=1.0, step=0.01,
        continuous_update=False,
        layout=Layout(width='45%')
    )
    iso_wgt.observe(bjh_plot, names='value')

    tab_wgts = [cor_wgt, thk_wgt, HBox([HTML(value='Relative Pressure'), iso_wgt])]
    tab_names = {0: 'BJH Correction', 1: 'Thickness Curve', 2: 'Isotherm'}
    tab = widgets.Tab(children=tab_wgts, _titles=tab_names)
    display(tab)

    # Make graph using pre defined widget values
    bjh_plot()

def print_vol(bjh_data, name='PythonVolumeData'):
    """Prints volume data that can be pasted into MicroActive's volume graph"""

    print(name + ' (A, cm3/g, cumulative)')
    for i in range(0, len(bjh_data.PoreRadii)):
        print(bjh_data.PoreRadii[i], bjh_data.cumV[i])
