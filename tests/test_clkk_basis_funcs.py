import os
import sys
sys.path.append("../")
import pk_func_fit_linear as pk2
import pk_func as pk1
import numpy as np
import matplotlib.pyplot as plt

alphas = np.arange(0.3,0.7,0.1)

pkcl1 = pk1.Cl_kk_supp()
pkcl2 = pk2.Cl_kk_supp()

pkcl1.alphas = alphas
pkcl2.alphas = alphas

Cl_o = pkcl1.get_Cl_kk(sup=False)
Cl_1 = pkcl1.get_Cl_kk(sup=True)
Cl_2 = pkcl2.get_Cl_kk()

cents,data_binned,cinv = pkcl1.make_mock_data(sup=True,alphas=alphas)

fig = plt.figure()
plt.loglog(pkcl1.ells,Cl_o,label="not sup, original")
plt.loglog(pkcl1.ells,Cl_1,label="sup, origianl")
plt.loglog(pkcl2.ells,np.sum(Cl_2,axis=1),label="not sup, new")
plt.loglog(cents,data_binned,label="sup, original,binned")
plt.legend()
plt.savefig("../plots/cl_kk_old_new.png")

k_bins = np.geomspace(5e-3,3e3,len(alphas)+1)
k_bins = np.round(k_bins,decimals = 3)

fig = plt.figure()
plt.plot(pkcl1.ells,Cl_o,label="not sup, original")
#plt.loglog(pkcl1.ells,Cl_1,label="sup, origianl")
for i in range(len(alphas)):
    lab = 'k_bin = ['+str(k_bins[i])+" ,"+str(k_bins[i+1])+"]"
    plt.plot(pkcl2.ells,Cl_2[:,i],label=lab)
#plt.loglog(cents,data_binned,label="sup, original,binned")
plt.xscale("log")
plt.legend()
plt.savefig("../plots/cl_kk_old_new_basis_funcs.png")


cl_arr = np.vstack((cents,data_binned))
np.savetxt("../mock_data/mock_cl_z_0.4_lin_fit.txt",cl_arr)
np.savetxt("../mock_data/mock_cinv_z_0.4_lin_fit.txt",cinv)
