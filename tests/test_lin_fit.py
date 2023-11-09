import sys
sys.path.append("../")
import pk_func_fit_linear as pk
import numpy as np

alphas_true = np.array([0.3])
pkcl = pk.Cl_kk_supp()
k_bins = np.array([4.5e-5,3e3])
pkcl.k_bins = k_bins
pkcl.n_alphas = len(k_bins)-1

print(pkcl.k_bins)
print(alphas_true)
cents,data_binned,cinv = pkcl.make_mock_data(alphas_true,sup_now=True)

cl_arr = np.vstack((cents,data_binned))
np.savetxt("../mock_data/mock_cl_z_0.4_lin_fit.txt",cl_arr)
np.savetxt("../mock_data/mock_cinv_z_0.4_lin_fit.txt",cinv)

pkcl.k_bins = k_bins

X,chisquare = pkcl.eval()
print("results with same bins as mock data")
print(X)
print(chisquare)

"""
pkcl.k_bins = k_bins*0.9
X,chisquare = pkcl.eval()
print("results with slightly diff bins as mock data")
print(X)
print(chisquare)
"""
