import os
import sys
sys.path.append("../")
import pk_func_fit_linear as pk2
import pk_func as pk1
import numpy as np
import matplotlib.pyplot as plt

alphas = np.array([0.5,0.3,0.1])
k_bins = np.array([4.5e-5,4.5e-3,1.0e-1,3.0e3])

#pkcl1 = pk1.Cl_kk_supp()
pkcl2 = pk2.Cl_kk_supp()

pkcl2.k_bins = k_bins

#pkcl1.alphas = alphas

#Cl_o = pkcl1.get_Cl_kk(sup=False)
#Cl_1 = pkcl1.get_Cl_kk(sup=True)
Cl_2 = pkcl2.get_Cl_kk()

cents,data_binned,cinv = pkcl2.make_mock_data(alphas,k_bins)

"""
fig = plt.figure()
plt.loglog(pkcl1.ells,Cl_o,label="not sup, original")
plt.loglog(pkcl1.ells,Cl_1,label="sup, original")
plt.loglog(pkcl2.ells,np.sum(Cl_2,axis=1),label="not sup, new")
plt.loglog(cents,data_binned,label="sup, new, binned")
#plt.loglog(cents,data_binned,label="sup, original,binned")
plt.legend()
plt.savefig("../plots/cl_kk_old_new.png")
"""
k_bins = np.round(k_bins,decimals = 6)

fig = plt.figure()
#plt.plot(pkcl1.ells,Cl_o,label="not sup, original")
#plt.loglog(pkcl1.ells,Cl_1,label="sup, origianl")
for i in range(len(alphas)):
    lab = 'k_bin = ['+str(k_bins[i])+" ,"+str(k_bins[i+1])+"]"
    plt.plot(pkcl2.ells,pkcl2.ells*Cl_2[:,i],label=lab)
#plt.loglog(cents,data_binned,label="sup, original,binned")
plt.xscale("log")
plt.ylabel("$lC{_l}^{\kappa \kappa}$")
plt.xlabel("l")
plt.legend()
plt.savefig("../plots/cl_kk_fid_basis_funcs.png")
plt.close(fig)


fig = plt.figure()
#plt.plot(pkcl1.ells,Cl_o,label="not sup, original")
#plt.loglog(pkcl1.ells,Cl_1,label="sup, origianl")
for i in range(len(alphas)):
    lab = 'k_bin = ['+str(k_bins[i])+" ,"+str(k_bins[i+1])+"]"
    plt.plot(pkcl2.ells,alphas[i]*pkcl2.ells*Cl_2[:,i],label=lab)
#plt.loglog(cents,data_binned,label="sup, original,binned")
plt.ylabel("$lC{_l}^{\kappa \kappa}$")
plt.xlabel("l")
plt.xscale("log")
plt.legend()
plt.savefig("../plots/cl_kk_sup_basis_funcs.png")
#cl_arr = np.vstack((cents,data_binned))
#np.savetxt("../mock_data/mock_cl_z_0.4_lin_fit.txt",cl_arr)
#np.savetxt("../mock_data/mock_cinv_z_0.4_lin_fit.txt",cinv)
