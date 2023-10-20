import time, sys, os
import numpy as np
import matplotlib.pyplot as plt
import dynesty
import pk_func as pk

cl = pk.Cl_kk_supp()
zsrc = 0.4
cl.zsrc = zsrc

rstate= np.random.default_rng(56101)

dsampler = dynesty.DynamicNestedSampler(cl.logp,cl.prior_transform,ndim=20,bound="multi",sample="rwalk",rstate=rstate)

dsampler.run_nested(maxiter=10)
dres = dsampler.results

print(dres)
