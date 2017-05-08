"""
Bokeh application for bjh pore distributions

To run navigate to /micromeritics/standalone_apps/
open cmd and run: bokeh serve bjh_bokeh_app.py
Open browser and navigate to: localhost:5006

Make sure the micromeritics package is installed
"""

# Graphing
from bokeh.models.widgets import RadioGroup, Div
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column, layout, widgetbox
from bokeh.plotting import figure
from bokeh.io import curdoc

# Calculations
import numpy as np

from micromeritics import isotherm_examples as ex, thickness, bjh

s = ex.silica_alumina()
prel = s.Prel
qads = s.Qads
APF = 9.53000
DCF = 0.0015468
adsorption = True

bjh_data = bjh.standard(prel, qads, APF, DCF, thickness.HarkinsJura(), adsorption)
r, v, a = bjh_data.PoreRadii, bjh_data.cumV, bjh_data.cumA

bjh_cds = ColumnDataSource(dict(r=r, v=v, a=a))
iso_cds = ColumnDataSource(dict(prel=prel, qads=qads))

# Testing with diffrerent options
tools1 = 'pan,box_select,wheel_zoom,reset'
tools2 = 'xpan,xbox_select,xwheel_zoom,reset'
tools3 = 'pan,box_select,box_zoom,reset'

def bjh_plot(title, x_label, y_label, x_log=True, tools=tools1):

    x_axis_type = "log" if x_log else "linear"

    graph = figure(plot_width=600, plot_height=300, tools=tools, toolbar_location='above',
        x_axis_type=x_axis_type, title=title, active_drag=None)

    graph.xaxis.axis_label = x_label
    graph.xaxis.axis_label_text_font_style = "normal"

    graph.yaxis.axis_label = y_label
    graph.yaxis.axis_label_text_font_style = "normal"

    graph.toolbar.logo=None

    return graph

# Volume plot
vol_plot = bjh_plot(
        title='Cumulative Volume', x_label='Pore Radius [A]',
        y_label='Pore Volume [cm3/g]', tools=tools1)

vol_plot.line('r', 'v', source=bjh_cds, line_width=2, color="red")
vol_plot.circle('r', 'v', source=bjh_cds, size=3, color="red")

# Area plot
area_plot = bjh_plot(
        title='Cumulative Area', x_label='Pore Radius [A]',
        y_label='Pore Area [m2/g]', tools=tools2)

area_plot.line('r', 'a', source=bjh_cds, line_width=2, color="blue")
area_plot.circle('r', 'a', source=bjh_cds, size=3, color="blue")

# Isotherm plot
iso_plot = bjh_plot(
        title='Isotherm', x_label='Relative Pressure (p/p0)',
        y_label='Quantity Adsorbed [cm3/g STP]', x_log=False, tools=tools3)

iso_plot.line('prel', 'qads', source=iso_cds, line_width=2, color="green")
iso_plot.circle('prel', 'qads', source=iso_cds, size=3, color="green")

def update_data(attrname, old, new):
    selected_cor = cor_wgt.active
    selected_thk = thk_wgt.active

    iso_indicies = np.array(iso_cds.selected['1d']['indices'])
    iso_indicies = np.sort(iso_indicies)

    if iso_indicies.size == 0:
        selected_prel = prel
        selected_qads = qads
    else:
        selected_prel = np.array([iso_cds.data["prel"][i] for i in iso_indicies])
        selected_qads = np.array([iso_cds.data["qads"][i] for i in iso_indicies])

    bjh_data = bjh_dict[selected_cor](selected_prel, selected_qads, APF, DCF, thk_dict[selected_thk], adsorption)

    bjh_cds.data = dict(r=bjh_data.PoreRadii, v=bjh_data.cumV, a=bjh_data.cumA)

# Widget dictionary
bjh_dict = {
    0: bjh.standard,
    1: bjh.kjs,
    2: bjh.faas
}

thk_dict = {
    0: thickness.KrukJaroniecSayari(),
    1: thickness.Halsey(),
    2: thickness.HarkinsJura(),
    3: thickness.BroekhoffDeBoer(),
    4: thickness.CarbonBlackSTSA()
}

# Widgets
cor_wgt = RadioGroup(labels=['Standard', 'Kruk-Jaroniec-Sayari', 'Faas'], active=0, width = 200)
cor_wgt.on_change('active', update_data)
cor_label = Div(text='''<b>BJH Correction</b>''')

thk_wgt = RadioGroup(labels=['Kruk-Jaroniec-Sayari', 'Halsey', 'Harkins and Jura',
        'Broekhoff-de Boer', 'Carbon Black STSA'], active=2, width = 200)
thk_wgt.on_change('active', update_data)
thk_label = Div(text='''<b>Thickness Curve</b>''')

empty_wgt = RadioGroup(labels=[], width=50, height = 200)
empty_label = Div(text='''<center><a href="https://github.com/Micromeritics/micromeritics/tree/master/documentation" target="_blank">BJH Doc</a></center>''')

iso_cds.on_change('selected', update_data)

# Set up layout
bjh_widgets = layout([
    [widgetbox(empty_label), widgetbox(cor_label), widgetbox(thk_label)],
    [widgetbox(empty_wgt), widgetbox(cor_wgt), widgetbox(thk_wgt)]], responsive=True)
bjh_layout = layout([[vol_plot, bjh_widgets], [area_plot, iso_plot]], responsive=True)

curdoc().add_root(bjh_layout)
curdoc().title = "BJH App"
