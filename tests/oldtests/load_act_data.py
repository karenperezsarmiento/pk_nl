import numpy as np
import sys
sys.path.append("../")

start = 1
end = -6
def standardize(ls,cls,trim_lmax,lbuffer=2,extra_dims="y"):
    cstart = int(ls[0])
    diffs = np.diff(ls)
    if not(np.all(np.isclose(diffs,1.))): raise ValueError("Multipoles are not spaced by 1")
    if not(cstart<=2): raise ValueError("Multipoles start at value greater than 2")
    nlen = trim_lmax+lbuffer
    cend = nlen - cstart
    if extra_dims=="xyy":
        out = np.zeros((cls.shape[0],nlen,nlen))
        out[:,cstart:,cstart:] = cls[:,:cend,:cend]
    elif extra_dims=="yy":
        out = np.zeros((nlen,nlen))
        out[cstart:,cstart:] = cls[:cend,:cend]
    elif extra_dims=="xy":
        out = np.zeros((cls.shape[0],nlen))
        out[:,cstart:] = cls[:,:cend]
    elif extra_dims=="y":
        out = np.zeros(nlen)
        out[cstart:] = cls[:cend]
    else:
        raise ValueError
    return out

d = {}

y = np.loadtxt('act_dr6/clkk_bandpowers_act.txt')
nbins_tot_act = y.size
d['full_data_binned_clkk_act'] = y.copy()
data_act = y[start:end].copy()
d['data_binned_clkk'] = data_act
nbins_act = data_act.size
binmat = np.loadtxt("act_dr6/binning_matrix_act.txt")
d['full_binmat_act']  = binmat.copy()
trim_lmax=2998
pells = np.arange(binmat.shape[1])
bcents = binmat@pells
ls = np.arange(binmat.shape[1])
d['binmat_act'] = standardize(ls,binmat[start:end,:],trim_lmax,extra_dims="xy")
d['bcents_act'] = bcents.copy()

fcov = np.loadtxt('act_dr6/covmat_act.txt')
d['full_act_cov'] = fcov.copy()
sel = np.s_[nbins_tot_act+end:nbins_tot_act]
cov = np.delete(np.delete(fcov,sel,0),sel,1)
sel = np.s_[:start]
cov = np.delete(np.delete(cov,sel,0),sel,1)

nbins = d['data_binned_clkk'].size
nsims_act = 792.
nsims = nsims_act
hartlap_correction = (nsims-nbins-2.)/(nsims-1.)
d['cov'] = cov
cinv = np.linalg.inv(cov) * hartlap_correction
d['cinv'] = cinv

print(d["binmat_act"])

