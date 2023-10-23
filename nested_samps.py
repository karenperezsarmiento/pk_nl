import time, sys, os
import numpy as np
import dynesty
import pk_func as pk

cl = pk.Cl_kk_supp()
zsrc = 0.39
cl.zsrc = zsrc

rstate= np.random.default_rng(56101)

dsampler = dynesty.DynamicNestedSampler(cl.logp,cl.prior_transform,ndim=2,bound="single",sample="rwalk",rstate=rstate)

dsampler.run_nested(print_progress=True)
dres = dsampler.results

print(dres)
