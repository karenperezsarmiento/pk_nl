import emcee
import numpy as np
import pk_func
import corner

pkcl = pk_func.Cl_kk_supp()
alphas = np.ones(20)
pos = alphas + 1e-4*np.random.randn(50,20)
nwalkers, ndim = pos.shape
print(nwalkers)
print(ndim)
sampler = emcee.EnsembleSampler(nwalkers,ndim,pkcl.log_probability)
sampler.run_mcmc(pos,5000,progress = True)
samples = sampler.get_chain()
labels = ["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7","alpha8","alpha9","alpha10","alpha11","alpha12","alpha13","alpha14","alpha15","alpha16","alpha17","alpha18","alpha19","alpha20"]
flat_samples = sampler.get_chain(discard=100,thin=15,flat=True)
fig = corner.corner(flat_samples,labels=labels)
fig.savefig("plots/emcee_corner_plot.png")
