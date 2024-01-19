import sys
import matplotlib.pyplot as plt
sys.path.append("../")
import pk_func_cmb as pk
import numpy as np
import scipy.interpolate as interp
import dynesty
from dynesty import plotting as dyplot
from matplotlib import rcParams
import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

"""

def bmatrix(a):
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    temp_string = np.array2string(a, formatter={'float_kind':lambda x: "{:.2e}".format(x)})
    lines = temp_string.replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)
"""

kb_dict = {
           #0:np.array([5e-5,5e-3,5e-2,0.5,3e3]),
           1:np.array([5e-5,5e-3,0.5,3e3]),
           2:np.array([5e-5,5e-2,0.5,3e3]),
           3:np.array([5e-5,5e-4,5e-2,0.5,3e3]),
           #4:np.array([5e-5,5e-4,5e-2,1e-1,5e-1,3e3]),
           #5:np.array([5e-5,5e-4,5e-2,1e-1,5e-1,1,3e3])
            }

for jjj in range(1,4):
    binning = jjj
    k_bins = kb_dict[binning]
    pkcl = pk.Cl_kk_supp()
    pkcl.cmb_source = True
    pkcl.k_bins = k_bins
    ndims = len(k_bins) - 3
    rstate = np.random.default_rng(56101)
    dsampler = dynesty.DynamicNestedSampler(pkcl.logp,pkcl.prior_transform,ndim=ndims,bound="multi",sample="auto",rstate=rstate)
    #dsampler.run_nested(nlive_init=500,nlive_batch=100,maxiter_init=1000,maxiter_batch=10,maxbatch=1,print_progress=True)
    dsampler.run_nested(nlive_init=500,nlive_batch=100,maxiter=1000,maxiter_batch=100,maxbatch=10,wt_kwargs={'pfrac': 1.0},print_progress=True)
    dres = dsampler.results
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
    labels = [r'$\alpha_1$', r'$\alpha_2$']
    fig, axes = dyplot.traceplot(dsampler.results, labels=labels, fig=plt.subplots(int(ndims), 2, figsize=(16, 12)))
    fig.tight_layout()
    pfn = "nested_plots/traceplot_binning_"+str(jjj)+".png"
    plt.savefig(pfn)
    fig, axes = dyplot.cornerplot(dres, show_titles=True,labels=labels,fig=plt.subplots(int(ndims), int(ndims), figsize=(15, 15)))
    cfn = "nested_plots/cornerplot_binning_"+str(jjj)+".png"
    plt.savefig(cfn)
