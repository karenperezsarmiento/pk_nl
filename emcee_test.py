import emcee
import numpy as np
import pk_func
import corner

pkcl = pk_func.Cl_kk_supp()
alphas = np.ones(2)
pos = alphas + 1e-4*np.random.randn(32,2)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers,ndim,pkcl.log_probability)
sampler.run_mcmc(pos,5000,progress = True)
samples = sampler.get_chain()
labels = ["alpha1","alpha2"]
flat_smaples = sampler.get_chain(discard=100,thin=15,flat=True)
fig = corner.corner(flat_samples,labels=labels)
fig.savefig("plots/emcee_corner_plot.png")
