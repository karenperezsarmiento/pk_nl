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

kb_dict = {0:np.array([5e-5,5e-3,5e-2,0.5,3e3]),
            1:np.array([5e-5,5e-3,0.5,3e3]),
            2:np.array([5e-5,5e-2,0.5,3e3]),
            3:np.array([5e-5,5e-4,5e-2,0.5,3e3]),
            4:np.array([5e-5,5e-4,5e-2,1e-1,5e-1,3e3]),
            5:np.array([5e-5,5e-4,5e-2,1e-1,5e-1,1,3e3])}

for jjj in range(0,6):
    binning = jjj
    k_bins = kb_dict[binning]
    pkcl = pk.Cl_kk_supp()
    pkcl.cmb_source = True
    pkcl.k_bins = k_bins
    X,chisquare,cov,corr = pkcl.eval()
    z = 0.01
    chi_from_z = interp.interp1d(pkcl.z_arr,pkcl.chi_arr)
    chi = chi_from_z(z)
    chi_arr = pkcl.chi_arr
    ells = np.linspace(0,700,5000)
    k = (ells + 0.5)/chi
    errs = np.sqrt(np.diag(cov))
    alphas_fit = X.reshape(len(k_bins)-3)

    cents = pkcl.d["cents"]
    n_samples = 10000
    s_pars = np.zeros((len(alphas_fit),n_samples))
    for i in range(len(errs)):
        s_pars[i,:] = np.random.normal(alphas_fit[i],errs[i],n_samples)

    s_pars = np.vstack((np.vstack((np.ones(n_samples),s_pars)),np.ones(n_samples)))
    alphas = np.append(np.append(np.array(1),alphas_fit),np.array(1))
    errs = np.append(np.append(np.array(0),errs),np.array(0))
    rel_errs = errs/alphas
    errs_act = np.sqrt(1/np.diag(pkcl.d["cinv"]))

    ckk_funcs_binned = pkcl.get_Cl_kk_funcs()
    A = np.zeros((len(pkcl.d["cents"]),len(ckk_funcs_binned)))
    for i,func in enumerate(ckk_funcs_binned):
        ckk_i_b = ckk_funcs_binned[i](pkcl.d["cents"])
        A[:,i] = ckk_i_b 

    ck_range = np.dot(s_pars.T,A.T)
    ck_min = np.min(ck_range,axis=0)
    ck_max = np.max(ck_range,axis=0)
    
    P_fid = pkcl.get_Pk_suppressed(k)
    P_fit = P_fid * alphas

    color_code = ["r","b","orange","g","purple","lime","yellow"]

    fig,ax = plt.subplots()
    ax.errorbar(cents,cents*pkcl.d["data_binned"],cents*errs_act,label="Data",linestyle="none")
    ax.plot(cents,cents*np.sum(A*alphas,axis=1),label="tot fit",c="r")
    ax.fill_between(cents,cents*ck_min,cents*ck_max,color="red",alpha=0.4)
    ax.plot(cents,cents*np.sum(A,axis=1),label="fiducial",c="k")
    ax.set_yscale("log")
    ax.set_ylim(1e-5,2.5e-5)
    ax.set_ylabel(r"$\ell C_{\ell}$")
    ax.set_xlabel(r"$\ell$")
    ax.legend()
    fig.tight_layout()
    plt.savefig("act_fit_plots/"+str(binning)+"_ckk.png")
    plt.close(fig)


    fig = plt.figure()
    for i in range(1,len(alphas)-1):
        mask = A[:,i]>0
        plt.plot(cents[mask],cents[mask]*alphas[i]*A[:,i][mask],label="Fit to ACT in k bins: ["+str(np.round(k_bins[i],5))+" ,"+str(np.round(k_bins[i+1],5))+ "]",c=color_code[i])
        plt.plot(cents[mask],cents[mask]*A[:,i][mask],c="k")
        si = s_pars[i,:].reshape((n_samples,1))
        ckr = si*A[:,i]
        cki_max = np.max(ckr,axis=0)
        cki_min = np.min(ckr,axis=0)
        plt.fill_between(cents[mask],cents[mask]*cki_min[mask],cents[mask]*cki_max[mask],color=color_code[i],alpha = 0.5)
    #plt.plot(cents,cents*A[:,0],label="Fixed k bin: ["+str(np.round(k_bins[0],5))+" ,"+str(np.round(k_bins[1],5))+"]",c=color_code[0]) 
    plt.yscale("log")
    #plt.ylim(1e-7,2e-5)
    plt.ylabel(r"$\ell C_{\ell}$")
    plt.xlabel(r"$\ell$")
    plt.legend()
    fig.tight_layout()
    plt.savefig("act_fit_plots/"+str(binning)+"_ckk_window.png")
    plt.close(fig)

    fig = plt.figure()
    for i in range(1,len(alphas)-1):
        mask = (k>k_bins[i])&(k<k_bins[i+1]) 
        plt.plot(k[mask],P_fid[:,i][mask],c="k")
        plt.plot(k[mask],P_fit[:,i][mask],label="Fit to ACT in k bins: ["+str(np.round(k_bins[i],5))+" ,"+str(np.round(k_bins[i+1],5))+ "]",c=color_code[i])
        kmin = k_bins[i]
        kmax = k_bins[i+1]
        k_e = np.geomspace(kmin,kmax,3)
        Pm = np.mean(P_fit[:,i][mask])*np.ones(3)
        e = Pm*rel_errs[i]
        plt.plot(k_e,Pm,c=color_code[i])
        plt.errorbar(k_e[1],Pm[1],e[1],c=color_code[i])
    mask = P_fit[:,0]>0
    #plt.plot(k[mask],P_fit[:,0][mask],label="Fixed k bin: ["+str(np.round(k_bins[0],5))+" ,"+str(np.round(k_bins[1],5))+"]",c=color_code[0])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e2,2e5)
    plt.xlim(k_bins[1],k_bins[-2])
    plt.ylabel(r"$P_m (k)$")
    plt.xlabel(r"$k$")
    plt.legend()
    fig.tight_layout()
    plt.savefig("act_fit_plots/"+str(binning)+"_Pk_window.png")
    plt.close(fig)

    fig = plt.figure()
    for i in range(1,len(alphas)-1):
        mask = P_fit[:,i]>0
        plt.plot(k[mask],np.ones(len(k))[mask]*alphas[i],c=color_code[i])
        plt.fill_between(k[mask],np.ones(len(k))[mask]*(alphas[i]-errs[i]),np.ones(len(k))[mask]*(alphas[i]+errs[i]),alpha=0.4,label=r"$k_i =$ ["+ str(k_bins[i])+" ,"+str(k_bins[i+1])+"] "+r"$\alpha$ = "+str(np.round(alphas[i],3))+r"$\pm$"+str(np.round(errs[i],5)),color=color_code[i])
    plt.ylim(0.97,1.3)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\alpha$")
    plt.legend()
    fig.tight_layout()
    plt.savefig("act_fit_plots/"+str(binning)+"_alphas_fit_act.png")
    plt.close(fig)
