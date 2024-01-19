import sys
sys.path.append("../")
import pk_func_shearmock as pk
import numpy as np


print("--------------")
print("Mock data")
alphas_true = np.array([0.5,0.3,0.1])
print("mock data sup params")
print(alphas_true)
pkcl = pk.Cl_kk_supp()
k_bins = np.array([4.5e-5,4.5e-3,0.1,3e3])
print("mock data k bins")
print(k_bins)

pkcl.cmb_source = False
cents,data_binned,cinv = pkcl.make_mock_data(alphas_true,k_bins)


cl_arr = np.vstack((cents,data_binned))
fname_mock_data = "../mock_data/mock_cl_z_0.4_lin_fit.txt"
fname_mock_cinv = "../mock_data/mock_cinv_z_0.4_lin_fit.txt"
np.savetxt(fname_mock_data,cl_arr)
np.savetxt(fname_mock_cinv,cinv)

print("--------------")

new_k_bins = k_bins
#new_k_bins = np.array([4.5e-5,4.5e-3,4.5e-2,0.1,3e3])
pkcl = pk.Cl_kk_supp()
pkcl.cmb_source = False
pkcl.k_bins = new_k_bins
pkcl.fname_mock_data = fname_mock_data
pkcl.fname_mock_cinv = fname_mock_cinv
 
X,chisquare,cov,corr = pkcl.eval()
print("Fitting results")
print("fitting k_bins")
print(new_k_bins)
print("inferred sup params")
print(X)
print("covmat")
print(cov)
print("chisquare")
print(chisquare)
print("corr mat")
print(corr)
"""
pkcl.k_bins = k_bins*0.9
X,chisquare = pkcl.eval()
print("results with slightly diff bins as mock data")
print(X)
print(chisquare)
"""
