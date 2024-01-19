import numpy as np
import time
import os
import pyfisher as pf
from scipy.interpolate import interp1d
import camb
from camb import model
from sacc import Sacc, standard_types
from scipy.stats import binned_statistic as binnedstat
from scipy.interpolate import interp1d
import sys
sys.path.append("../")
import act_dr6_lenslike as act
from cobaya.likelihood import Likelihood


class bin1D:
    def __init__(self, bin_edges):
        self.update_bin_edges(bin_edges)
  
    def update_bin_edges(self,bin_edges):
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1
        self.cents = (self.bin_edges[:-1]+self.bin_edges[1:])/2.
        self.bin_edges_min = self.bin_edges.min()
        self.bin_edges_max = self.bin_edges.max()

    def bin(self,ix,iy,stat=np.nanmean):
        x = ix.copy()
        y = iy.copy()
        y[x<self.bin_edges_min] = 0
        y[x>self.bin_edges_max] = 0
        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=stat)[0]
        return self.cents,bin_means

class Cl_kk_supp(Likelihood):
    H0 = 68.3 #actdr6
    ombh2 = 0.022
    omch2 = 0.122
    ns = 0.965
    k_min = 1e-4
    k_max = 100
    zmax = 5000
    zsrc = 0.4
    ells = np.arange(2,5000)
    z_arr = np.arange(0.001,50,0.01)
    ez = np.arange(0.001,6,0.01)
    lmin = 5
    lmax = 4999
    nbins = 40
    shape_std = 0.3
    ngal_arcmin2 = 15
    data_file = "hsc_y3_fourier_space_data_vector.sacc"
    fsky = 1.
    d = {}
    z_cmb = 1100
    mock_data = False
    cmb_source = True
    fname_mock_data = "/home3/kaper/pk_nl/mock_data/mock_cl_cmb_lin_fit.txt"
    fname_mock_cinv = "/home3/kaper/pk_nl/mock_data/mock_cinv_cmb_lin_fit.txt"
    n_alphas = 3
    k_bins = np.array([4.5e-5,4.5e-3,0.1,3e3])

    def initialize(self):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        pars.InitPower.set_params(ns=self.ns)
        self.results = camb.get_background(pars)
        self.pars = pars
        self.chi_arr = self.results.comoving_radial_distance(self.z_arr)
        self.chi_ez = self.results.comoving_radial_distance(self.ez)
        self.chi_cmb = self.results.comoving_radial_distance(self.z_cmb)
        self.c = 3e5
        self.H_z = self.results.hubble_parameter(self.z_arr)
        self.PK = camb.get_matter_power_interpolator(self.pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=self.k_max,zmax=self.zmax)
        self.bin_edges = np.geomspace(self.lmin,self.lmax,self.nbins)
        self.binner = bin1D(self.bin_edges)
        self.load_data()

    def get_requirements(self):
        return {'alpha1':None,'alpha2':None}

    def load_data(self):
        if self.mock_data == True:
            cents,data_binned = np.loadtxt(self.fname_mock_data)
            cinv = np.loadtxt(self.fname_mock_cinv)
        else:
            data = self.load_real_data()
            cents = data["bcents_act"]
            data_binned = data["data_binned_clkk"]
            cinv = data["cinv"]
            binmat = data["binmat_act"]
            self.d["binmat_act"] = binmat
        self.d["cents"] = cents
        self.d['cinv'] = cinv
        self.d["data_binned"] = data_binned

    def load_real_data(self):
        variant = "act_baseline"
        lens_only = False
        no_like_corrections = False
        apply_hartlap = True
        mock = False
        nsims_act = 792. 
        nsims_planck = 400.
        trim_lmax = 2998
        scale_cov = None
        data = act.load_data(variant=variant,lens_only=lens_only,like_corrections=not(no_like_corrections),apply_hartlap=apply_hartlap,mock=mock,nsims_act=nsims_act,nsims_planck=nsims_planck,trim_lmax=trim_lmax,scale_cov=scale_cov)
        return data

    def make_mock_data(self,alphas,k_bins):
        self.alphas = alphas
        self.k_bins = k_bins
        self.n_alphas = len(self.alphas)
        cl = self.get_Cl_kk()
        cl = cl*self.alphas
        cl_out = np.sum(cl,axis=1)
        if self.cmb_source:
            nl_ells,nl = pf.get_lensing_nl("so_goal")
            nls_dict = {'kk': interp1d(nl_ells,nl,bounds_error=True)}
        else:
            nls_dict = {'kk': lambda x: x*0+self.shape_std**2/(2.*self.ngal_arcmin2*1.18e7)}
        noise_cov = nls_dict["kk"](self.ells)
        self.noise_cov = noise_cov
        cls_dict = {"kk":interp1d(self.ells,cl_out)}
        cents,data_binned = self.binner.bin(self.ells,cl_out)
        cov = pf.gaussian_band_covariance(self.bin_edges,['kk'],cls_dict,nls_dict,interpolate=False)[:,0,0] / self.fsky
        cinv = np.diag(1./cov)
        return cents,data_binned,cinv

    def get_theo(self):
        cl_kappa = self.get_Cl_kk()
        self.d["cents"],self.d["theory_binned"] = self.binner.bin(self.ells,cl_kappa)

    """
    def load_data(self):
        f = "data/"+self.data_file
        d =  Sacc.load_fits(f)
        cls_list = ['cl_eb','cl_be','cl_bb','cl_0b','cl_00','cl_0e']
        for k in cls_list:
            d.remove_selection(data_type=k)
        cl_data = {}
        for t1,t2 in d.get_tracer_combinations():
            if (t1==t2):
                l,cl,cov = d.get_ell_cl('cl_ee',t1,t2,return_cov=True)
                cl_data[t1] = {
                    "l": l,
                    "cl":cl
                }
        #print(cl_data.keys())
        return cl_data
    """

    def get_pure_Pk(self,k):
        return self.PK.P(self.z_arr,k,grid=False)

    def get_Pk_suppressed(self,k):
        Pk = self.get_pure_Pk(k)
        inds = np.digitize(k,self.k_bins)-1
        Pk_sup_out = np.zeros((len(Pk),self.n_alphas))
        for ind in range(self.n_alphas):
            inds_mask = (inds==ind)
            Pk_sup = inds_mask * Pk
            Pk_sup_out[:,ind] = Pk_sup
        return Pk_sup_out

    def get_window_kk(self):
        h = self.H0/100
        omega_m = (self.ombh2/h**2)+(self.omch2/h**2)
        f = (3./2.)*omega_m * self.H0**2 * (self.chi_arr/self.c) * (1+self.z_arr)/self.H_z
        if self.cmb_source == False:
            dndz = np.exp(-(self.ez-self.zsrc)**2/(0.1**2))
            dndz = dndz/np.trapz(dndz,self.ez)
            integrand = dndz[None,:] * (self.chi_ez[None,:] - self.chi_arr[:,None])/self.chi_ez[None,:]
            for i in range(integrand.shape[0]):
                integrand[i][self.ez<self.z_arr[i]] = 0 
            integral = np.trapz(integrand,self.ez,axis=-1)
            window_kk = f*integral
        else:
            integral = (self.chi_cmb - self.chi_arr)/self.chi_cmb
            window_kk = f*integral
        return window_kk

    def get_Cl_kk(self):
        window_kk = self.get_window_kk()
        C_kk = np.zeros((self.ells.shape[0],self.n_alphas))
        for i in range(len(self.ells)):
            k = (self.ells[i] + 0.5)/self.chi_arr
            Pk_sup_out = self.get_Pk_suppressed(k)
            integrand = (1/self.c)*(self.H_z[:,None] / self.chi_arr[:,None]**2 )*window_kk[:,None]**2 * Pk_sup_out
            C_kk[i,:] = np.trapz(integrand,self.z_arr[:,None],axis=0)
        return C_kk
    
    def get_Cl_kk_funcs(self):
        f = self.get_Cl_kk()
        print(f.shape)
        f_b = np.zeros((self.d["cents"].shape[0],f.shape[1]))
        for i in range(f.shape[1]):
            if self.mock_data:
                cs,f_b[:,i] = self.binner.bin(self.ells,f[:,i])
            else:
                cl_stand = act.standardize(self.ells,f[:,i],trim_lmax=2998)
                f_b[:,i] = self.d["binmat_act"] @ cl_stand
                cs = self.d["cents"]
        funcs = [interp1d(cs,f_b[:,j]) for j in range(f_b.shape[1])]
        return funcs 
    
    def cov2corr(self,mat):
        diags = np.diagonal(mat).T
        xdiags = diags[:,None,...]
        ydiags = diags[None,:,...]
        corr = mat/np.sqrt(xdiags*ydiags)
        return corr

    def logp(self,**kwargs):
        alphas = np.array([kwargs[p] for p in ["alpha1","alpha2"]])
        ### DELETE HARD CODED kbins later
        self.k_bins = np.array([5e-5,5e-3,5e-2,0.5,3e3])
        alphas_2 = np.append(np.append(np.array(1),alphas),np.array(1))
        self.alphas = alphas_2
        print(self.alphas)
        self.n_alphas = len(self.alphas)
        assert len(self.k_bins) == self.n_alphas + 1
        c_kk = self.get_Cl_kk()
        sup_c_kk = self.alphas*c_kk
        c_theo_u = np.sum(sup_c_kk,axis=1)
        if self.mock_data:
            cents,self.d["theory_binned"] = self.binner.bin(self.ells,c_theo_u)
        else:
            cl_theo_stand = act.standardize(self.ells,c_theo_u,trim_lmax=2998)
            self.d["theory_binned"] = self.d["binmat_act"] @ cl_theo_stand 
        #assert (self.d["cents"]==cents).all()
        delta = self.d["data_binned"] - self.d["theory_binned"]
        lnlike =  -0.5 * np.dot(delta,np.dot(self.d["cinv"],delta))
        return lnlike

    def prior_transform(self,ualphas):
        alphas = 0.3*ualphas + 1. #tighten prior to U[0.7,1.3]
        return alphas
        
        
    def fit_linear_model(self,x,y,cinv,funcs):
        y = y.reshape((y.size,1))
        A = np.zeros((y.size,len(funcs)))
        for i,func in enumerate(funcs):
            A[:,i] = func(x)
        CA = np.dot(cinv,A)
        cov = np.linalg.inv(np.dot(A.T,CA))
        Cy = np.dot(cinv,y)
        b = np.dot(A.T,Cy)
        X = np.dot(cov,b)
        YAX = (y-np.dot(A,X))
        chisquare = np.dot(YAX.T,np.dot(cinv,YAX))
        corr = self.cov2corr(cov)
        return X, chisquare,cov,corr

    def eval(self):
        self.load_data()
        self.n_alphas = len(self.k_bins)-1
        x = self.d["cents"]
        y = self.d["data_binned"]
        cinv = self.d["cinv"]
        funcs = self.get_Cl_kk_funcs()
        c_k_o = funcs[0](x)
        c_k_f = funcs[-1](x)
        y = y - c_k_o - c_k_f #fit everything except the first and last bin
        X,chisquare,cov,corr= self.fit_linear_model(x,y,cinv,funcs[1:-1])
        return X, chisquare,cov,corr
        
