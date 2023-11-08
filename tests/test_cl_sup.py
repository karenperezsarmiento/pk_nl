import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import matplotlib.pyplot as plt

pkcl = pk.Cl_kk_supp()
pkcl.zsrc = 0.4
alphas = np.linspace(0,1,20)
pkcl.alphas = alphas
ell = 1000
k = (ell + 0.5)/pkcl.chi_arr
sup = pkcl.get_Pk_suppressed(k)
p = pkcl.get_pure_Pk(k)

fig = plt.figure()
plt.plot(k,sup,label="sup")
plt.plot(k,p,label="normal")
plt.legend()
plt.yscale("log")
plt.savefig("../plots/pk_sup_vs_reg.png")
plt.close(fig)

cents,data_binned,cinv = pkcl.make_mock_data(sup=True,alphas=alphas)
cl_arr = np.vstack((cents,data_binned))
np.savetxt("../mock_data/mock_sup_cl_0.4.txt",cl_arr)
np.savetxt("../mock_data/mock_sup_cinv_0.4.txt",cinv)
