import sys
sys.path.append("../")
import pk_func_fit_linear as pk2
import pk_func as pk
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt

pkcl = pk2.Cl_kk_supp()
z = 1.
chi_from_z = interp.interp1d(pkcl.z_arr,pkcl.chi_arr)
chi = chi_from_z(z)
ells = np.round(np.linspace(0,10000,5000))
k = (ells + 0.5)/chi
p = pkcl.get_Pk_suppressed(k)

fig = plt.figure()
plt.loglog(k,p)
plt.savefig("../plots/pk_binned.png")
