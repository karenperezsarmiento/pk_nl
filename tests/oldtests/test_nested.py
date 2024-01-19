import time, sys, os

# basic numeric setup
import numpy as np

# inline plotting
#%matplotlib inline

# plotting
import matplotlib
from matplotlib import pyplot as plt
from dynesty import plotting as dyplot

# seed the random number generator
rstate= np.random.default_rng(56101)

# re-defining plotting defaults
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 30})

import dynesty

# truth
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# generate mock data
N = 50
x = np.sort(10 * rstate.uniform(size=N))
yerr = 0.1 + 0.5 * rstate.uniform(size=N)
y_true = m_true * x + b_true
y = y_true + np.abs(f_true * y_true) * rstate.normal(size=N)
y += yerr * rstate.normal(size=N)

# log-likelihood
def loglike(theta):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
    
    return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# prior transform
def prior_transform(utheta):
    um, ub, ulf = utheta
    m = 5.5 * um - 5.
    b = 10. * ub
    lnf = 11. * ulf - 10.
    
    return m, b, lnf


truths = [m_true, b_true, np.log(f_true)]
labels = [r'$m$', r'$b$', r'$\ln f$']
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=3,
                                        bound='multi', sample='rwalk', rstate=rstate)
dsampler.run_nested(checkpoint_file="test_linear_regression.txt")
dres = dsampler.results
samples = dres["samples"]
logvol = dres["logvol"]
weights = dres.importance_weights()
fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                             fig=plt.subplots(3, 2, figsize=(16, 12)))
fig.tight_layout()
plt.savefig("traceplot.png")
