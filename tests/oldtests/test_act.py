import sys
import matplotlib.pyplot as plt
sys.path.append("../")
import pk_func_cmb as pk
import numpy as np
import scipy.interpolate as interp

#plt.rcParams['text.usetex'] = True
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

"""

def bmatrix(a):
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    temp_string = np.array2string(a, formatter={'float_kind':lambda x: "{:.2e}".format(x)})
    lines = temp_string.replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)
"""
kb_dict = {0:np.array([4.5e-5,5e-3,4.5e-2,0.1,3e3]),1:np.array([4.5e-5,5e-4,5e-3,5e-2,5e-1,5e0,5e1,3e3]),2:np.array([4.5e-5,5e-3,9e-3,5e-2,9e-3,5e-2,9e-2,5e-1,9e-1,1,3e3]),3:np.array([4.5e-5,5e-3,0.1,3e3])}
k_bins = np.array([4.5e-5,4.5e-3,4.5e-2,0.1,3e3])
pkcl = pk.Cl_kk_supp()
pkcl.cmb_source = True
pkcl.k_bins = k_bins
X,chisquare,cov,corr = pkcl.eval()
print("Fitting results")
print("fitting k_bins")
print(k_bins)
print("inferred sup params")
print(X)
print("covmat")
print(cov)
print("chisquare")
print(chisquare)
print("corr mat")
print(corr)
z = 0.01
chi_from_z = interp.interp1d(pkcl.z_arr,pkcl.chi_arr)
chi = chi_from_z(z)
chi_arr = pkcl.chi_arr
ells = np.linspace(0,2000,5000)
k = (ells + 0.5)/chi
errs = np.diag(cov)**2
#print("errs alphas")
print(errs)
alphas_fit = X.reshape(len(k_bins)-1)
P_fit = alphas_fit*pkcl.get_Pk_suppressed(k)
P_fid = pkcl.get_Pk_suppressed(k)
#print("rel errs are")
rel_errs = errs/alphas_fit

#print(rel_errs)

fig = plt.figure()
for i in range(1,len(alphas_fit)):
    mask = P_fit[:,i]>0
    plt.plot(k[mask],np.ones(len(k))[mask]*alphas_fit[i])
    plt.fill_between(k[mask],np.ones(len(k))[mask]*(alphas_fit[i]-10*errs[i]),np.ones(len(k))[mask]*(alphas_fit[i]+10*errs[i]),alpha=0.4,label=r"$k_i =$ ["+ str(k_bins[i])+" ,"+str(k_bins[i+1])+"] "+r"$\alpha$ = "+str(np.round(alphas_fit[i],3))+r"$\pm$"+str(np.round(errs[i],5)))
plt.yscale("log")
plt.xscale("log")
plt.ylim(0.7,1.3)
plt.xlim(1e-2,3e-1)
plt.xlabel(r"$k$")
plt.legend()
plt.savefig("act_fit_plots/alphas_fit_act.png")    
plt.close(fig)


fig = plt.figure()
for i in range(1,len(alphas_fit)):
    mask = P_fit[:,i]>0
    plt.plot(k[mask],P_fit[:,i][mask],label="Fitted $P_m$ in bin: ["+str(k_bins[i])+" ,"+str(k_bins[i+1]))
    plt.fill_between(k[mask],P_fit[:,i][mask]*(1+10*rel_errs[i]),P_fit[:,i][mask]*(1-10*rel_errs[i]),alpha=0.2)
    plt.plot(k[mask],P_fid[mask],label="Fitted $P_m$ in bin: ["+str(k_bins[i])+" ,"+str(k_bins[i+1]))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel(r"$P_m$")
plt.xlim(1e-2,3e-1)
plt.ylim(1e2,1e5)
#plt.legend()
plt.title(r"$P_m(k)$ at redshift $z=0$")
plt.savefig("act_fit_plots/P_k_windows_act.png")
plt.close(fig)

abs_errs = rel_errs*P_fit
tot_err = np.sum(abs_errs,axis=1)
fig,ax1 = plt.subplots()
ax1.plot(k,np.sum(P_fit,axis=1),label="Fitted $P_m$ ACT")
ax1.fill_between(k,np.sum(P_fit,axis=1)-tot_err,np.sum(P_fit,axis=1)+tot_err)
ax1.plot(k,np.sum(P_fid,axis=1),label="Fiducial $P_m$")
ax1.legend()
ax1.set_title(r"$P_m(k)$ at redshift $z=0$")
ax1.set_xlim(1e-2,3e-1)
ax1.set_ylim(1e2,1e5)
ax1.set_xlabel("k")
ax1.set_ylabel(r"$P_m$")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax2 = ax1.twinx()
for i in range(1,len(alphas_fit)):
    mask = P_fit[:,i]>0
    ax2.fill_between(k[mask],(k*0+1-rel_errs[i])[mask],(k*0+1+rel_errs[i])[mask],alpha=0.5)
ax2.set_yscale("log")
ax2.set_ylabel("Relative error in $P_m(k)$")
fig.tight_layout()
plt.savefig("act_fit_plots/P_k_fit_act.png")
plt.close(fig)

pkcl.logp(alphas_fit)

errs_act = np.sqrt(1/np.diag(pkcl.d["cinv"]))
ckk_funcs_binned = pkcl.get_Cl_kk_funcs()
T_err_l = np.zeros((len(pkcl.d["cents"]),len(ckk_funcs_binned)))
A = np.zeros((len(pkcl.d["cents"]),len(ckk_funcs_binned)))
for i,func in enumerate(ckk_funcs_binned):
    ckk_i_b =ckk_funcs_binned[i](pkcl.d["cents"])
    A[:,i] = ckk_i_b
    #maskckk = 
    rel_act_i = errs_act / ckk_i_b
    print(rel_act_i)
    tot_err_i_l = np.sqrt(rel_errs[i]**2 + rel_act_i**2)
    T_err_l[:,i] = tot_err_i_l
    

#print(T_err_l.shape)
tot_errs_l = np.sum(T_err_l**2,axis=1) 
#print(tot_errs_l.shape)
fig,ax = plt.subplots()
ax.errorbar(pkcl.d["cents"],pkcl.d["cents"]*np.sum(A*alphas_fit,axis=1),pkcl.d["cents"]*tot_errs_l,label="Fit to ACT",linestyle="none",elinewidth=10)
pkcl.logp(np.ones(len(k_bins)-1))
ax.scatter(pkcl.d["cents"],pkcl.d["cents"]*pkcl.d["theory_binned"],label="Fiducial",c="k")
ax.errorbar(pkcl.d["cents"],pkcl.d["cents"]*pkcl.d["data_binned"],pkcl.d["cents"]*errs_act,label="ACT DR6 $C_{\ell}^{\kappa \kappa}$",linestyle="none")
ax.set_yscale("log")
ax.legend()
ax.set_ylabel(r"$\ell C_{\ell}$")
plt.ylim(9e-6,2.1e-5)
ax.set_xlabel(r"$\ell$")
#ax.text(0.05,0.95,corr)
#ax.text(0.05,0.05,k_bins)
fig.tight_layout()
plt.savefig("act_fit_plots/ckk_total_binned.png")
plt.close(fig)

fig,ax = plt.subplots()
for i in range(1,len(alphas_fit)):
    ax.plot(pkcl.d["cents"],pkcl.d["cents"]*alphas_fit[i]*A[:,i],label = "Fit to ACT in k bin : ["+str(np.round(k_bins[i],4))+" ,"+str(np.round(k_bins[i+1],4))    + "]",alpha = 0.5)
    #ax.errorbar(pkcl.d["cents"],pkcl.d["cents"]*alphas_fit[i]*A[:,i],pkcl.d["cents"]*T_err_l[:,i],label = "Fit to ACT in k bin : ["+str(np.round(k_bins[i],4))+" ,"+str(np.round(k_bins[i+1],4))+ "]",linestyle="none",alpha = 0.5)
    ax.plot(pkcl.d["cents"],pkcl.d["cents"]*A[:,i],c="k")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$\ell C_{\ell}$")
ax.set_xlabel(r"$\ell$")
ax.set_ylim(1e-7,2e-5)
ax.legend()
fig.tight_layout()
plt.savefig("act_fit_plots/ckk_windows_binned.png")
plt.close(fig)

ckk = pkcl.get_Cl_kk()
ckk_fit = alphas_fit*ckk
fig,ax = plt.subplots()
for i in range(len(alphas_fit)):
    ax.plot(pkcl.ells,pkcl.ells*ckk[:,i],c="k")
    ax.plot(pkcl.ells,pkcl.ells*ckk_fit[:,i],label="Fit to ACT in k bins: ["+str(np.round(k_bins[i],4))+" ,"+str(np.round(k_bins[i+1],4))+ "]")
    #ax.fill_between(pkcl.ells,pkcl.ells*(ckk_fit[:,i]*(1-rel_errs[i])),pkcl.ells*(ckk_fit[:,i]*(1+rel_errs[i])),alpha=0.5)
ax.plot(pkcl.ells,pkcl.ells*np.sum(ckk,axis=1),c="k")
ax.plot(pkcl.ells,pkcl.ells*np.sum(ckk_fit,axis=1),label="Fit to ACT, total")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$\ell C_{\ell}$")
ax.set_xlabel(r"$\ell$")
ax.legend()
fig.tight_layout()
plt.savefig("act_fit_plots/ckk_total_windows_not_binned.png")
plt.close(fig)

fig,ax = plt.subplots()
ax.plot(pkcl.ells,pkcl.ells*np.sum(ckk,axis=1),c="k")
ax.plot(pkcl.ells,pkcl.ells*np.sum(ckk_fit,axis=1),label="Fit to ACT, total")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$\ell C_{\ell}$")
ax.set_xlabel(r"$\ell$")
plt.savefig("act_fit_plots/ckk_total_not_binned.png")
plt.close(fig)
