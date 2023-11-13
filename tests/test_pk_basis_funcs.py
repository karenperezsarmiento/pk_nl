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
chi_arr = pkcl.chi_arr
ells = np.round(np.linspace(0,10000,5000))
k = (ells + 0.5)/chi

k_bins = np.array([4.5e-5,4.5e-3,1.0e-1,3.0e+3])
#pkcl.n_alphas = len(pkcl.k_bins) - 1
alphas = np.array([0.5,0.3,0.1])

cents,data_binned,cinv = pkcl.make_mock_data(alphas,k_bins)

print(pkcl.n_alphas)
print(pkcl.alphas)
print(pkcl.k_bins)

X,chisquare,cov,corr = pkcl.eval()

errs_arr = np.diag(cov)**2

print(errs_arr)

p_sup = alphas*pkcl.get_Pk_suppressed(k)

pkcl.alphas = np.array([1.,1.,1.])
pkcl.n_alphas = len(pkcl.alphas)
pkcl.k_bins = k_bins

p_fid = pkcl.get_Pk_suppressed(k)

rat = np.divide(p_sup,p_fid, out = np.zeros_like(p_sup), where=p_fid!=0)

fig = plt.figure()
for i in range(rat.shape[1]):
    inds = (rat[:,i]>0.)
    plt.plot(k[inds],rat[inds,i])
    plt.fill_between(k[inds],rat[inds,i]-errs_arr[i],rat[inds,i]+errs_arr[i],alpha=0.2)
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("$P_{supp}/P_{fid}$")
plt.ylim(0,1)
plt.savefig("../plots/pk_alpha_errs.png")

fig = plt.figure()
plt.loglog(k,p_fid,label="Pk fiducial")
plt.loglog(k,p_sup,label="Pk suppressed")
plt.legend()
plt.savefig("../plots/pk_binned.png")
