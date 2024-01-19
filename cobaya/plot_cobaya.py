import numpy as np
import getdist
import matplotlib.pyplot as plt
import getdist.plots as gdplt
from getdist.mcsamples import loadMCSamples

gd_sample = loadMCSamples("/home3/kaper/pk_nl/pk_cobaya")

names = ["alpha1","alpha2"]
labels = names
gdplot = gdplt.get_subplot_plotter()
gdplot.triangle_plot(gd_sample, names, filled=True)
gdplot.export("/home3/kaper/pk_nl/cobaya/triangle_cobaya_binning0.png")
