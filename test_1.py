import pk_func as pk
import numpy as np
#from pk_func import Cl_kk_supp as cl
import matplotlib.pyplot as plt
import time

zsrc_arr = [0.2,0.4,0.6,0.8,1.0]
pkcl = pk.Cl_kk_supp()
for i in range(len(zsrc_arr)):
    pkcl.zsrc = zsrc_arr[i]
    alphas = np.ones(20)
    logp = pkcl.logp(alphas)
    print(logp)


