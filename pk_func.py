import numpy as np
import time
import os
import pyfisher as pf
from scipy.interpolate import interp1d
import camb
from camb import model
from sacc import Sacc, standard_types
from scipy.stats import binned_statistic as binnedstat

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

class Cl_kk_supp:
    H0 = 67.5
    ombh2 = 0.022
    omch2 = 0.122
    ns = 0.965
    k_min = 1e-4
    k_max = 100
    zmax = 5000
    zsrc = 0.4
    ells = np.arange(0,10000)
    z_arr = np.arange(0.001,50,0.01)
    ez = np.arange(0.001,6,0.01)
    lmin = 10
    lmax = 9000
    nbins = 40
    shape_std = 0.3
    ngal_arcmin2 = 15
    data_file = "hsc_y3_fourier_space_data_vector.sacc"
    fsky = 1.
    d = {}
    mock_data = True
    fname_mock_data = "/home3/kaper/pk_nl/mock_data/mock_cl_0.4.txt"
    fname_mock_cinv = "/home3/kaper/pk_nl/mock_data/mock_cinv_0.4.txt"
    n_alphas = 2
    alphas = np.ones(n_alphas)

    def __init__(self):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        pars.InitPower.set_params(ns=self.ns)
        self.results = camb.get_background(pars)
        self.pars = pars
        self.chi_arr = self.results.comoving_radial_distance(self.z_arr)
        self.chi_ez = self.results.comoving_radial_distance(self.ez)
        self.c = 3e5
        self.H_z = self.results.hubble_parameter(self.z_arr)
        self.PK = camb.get_matter_power_interpolator(self.pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=self.k_max,zmax=self.zmax)
        self.bin_edges = np.geomspace(self.lmin,self.lmax,self.nbins)
        self.binner = bin1D(self.bin_edges)
        if self.mock_data == True:
            cents,data_binned = np.loadtxt(self.fname_mock_data)
        else:
            data = self.load_data()
            cents,data_binned = self.binner.bin(data["wl_0"]["l"],data["wl_0"]["cl"])
        self.d['cinv'] = np.loadtxt(self.fname_mock_cinv)
        self.d["data_binned"] = data_binned
        

    def make_mock_data(self,sup):
        cl = self.get_Cl_kk(sup=sup)
        nls_dict = {'kk': lambda x: x*0+self.shape_std**2/(2.*self.ngal_arcmin2*1.18e7)}
        cls_dict = {"kk":interp1d(self.ells,cl)}
        cents,data_binned = self.binner.bin(self.ells,cl)
        cov = pf.gaussian_band_covariance(self.bin_edges,['kk'],cls_dict,nls_dict,interpolate=False)[:,0,0] / self.fsky
        cinv = np.diag(1./cov)
        return cents,data_binned,cinv

    def get_theo(self):
        cl_kappa = self.get_Cl_kk(sup=True)
        self.d["cents"],self.d["theory_binned"] = self.binner.bin(self.ells,cl_kappa)

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

    def get_pure_Pk(self,k):
        return self.PK.P(self.z_arr,k,grid=False)

    def get_Pk_suppressed(self,k):
        Pk = self.get_pure_Pk(k)
        k_bins = np.geomspace(5e-5,3e3,self.n_alphas)
        inds = np.digitize(k,k_bins)-1
        alphas_k = [self.alphas[j] for j in inds]
        Pk_sup = alphas_k*Pk
        return Pk_sup

    def get_window_kk(self):
        dndz = np.exp(-(self.ez-self.zsrc)**2/(0.1**2))
        dndz = dndz/np.trapz(dndz,self.ez)
        h = self.H0/100
        omega_m = (self.ombh2/h**2)+(self.omch2/h**2)
        f = (3./2.)*omega_m * self.H0**2 * (self.chi_arr/self.c) * (1+self.z_arr)/self.H_z
        integrand = dndz[None,:] * (self.chi_ez[None,:] - self.chi_arr[:,None])/self.chi_ez[None,:]
        for i in range(integrand.shape[0]):
            integrand[i][self.ez<self.z_arr[i]] = 0 
        integral = np.trapz(integrand,self.ez,axis=-1)
        window_kk = f*integral
        return window_kk

    def get_Cl_kk(self,sup=True):
        window_kk = self.get_window_kk()
        C_kk = np.zeros(self.ells.shape)
        for i in range(len(self.ells)):
            k = (self.ells[i] + 0.5)/self.chi_arr
            if sup:
                Pk_sup = self.get_Pk_suppressed(k)
            else:
                Pk_sup = self.get_pure_Pk(k)
            integrand = (1/self.c)*(self.H_z / self.chi_arr**2 )*window_kk**2 * Pk_sup
            C_kk[i] = np.trapz(integrand,self.z_arr)
        return C_kk

    def logp(self,alphas):
        self.alphas = alphas
        self.get_theo()
        delta = self.d["data_binned"] - self.d["theory_binned"]
        lnlike = -0.5 * np.dot(delta,np.dot(self.d["cinv"],self.d["theory_binned"]))
        return lnlike

    def prior_transform(self,ualphas):
        alphas = -0.5*ualphas - 1
        return alphas
