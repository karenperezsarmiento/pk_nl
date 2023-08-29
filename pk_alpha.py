import numpy as np
import os
import pyfisher as pf
from scipy.interpolate import interp1d
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

class Generate_Pk_Alpha:
    zsrc = 620.7057
    zeval = 0.2
    H0 = 67.5
    ombh2 = 0.022
    omch2 = 0.122
    ns = 0.965
    k_min = 1e-3
    k_max = 10
    nbins = 30
    kmax = 15

    def __init__(self):
        self.k_cents,self.Pk_sup = self.get_Pk_suppressed()

    def get_Pk_suppressed(self):
        bin_edges = np.geomspace(self.k_min,self.k_max,self.nbins)
        binner = bin1D(bin_edges) 
        self.k = np.exp(np.log(10)*np.linspace(-4,2,500))
        Pk = self.get_pure_Pk()
        k_cents,Pk_binned = binner.bin(ix=self.k,iy=Pk)
        alphas = self.get_alphas(k_cents)
        Pk_alphas = alphas*Pk_binned
        return k_cents,Pk_alphas

    def get_pure_Pk(self):
        import camb
        from camb import model
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        pars.InitPower.set_params(ns=self.ns)
        results = camb.get_background(pars)
        PK = camb.get_matter_power_interpolator(pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=self.kmax,var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=self.zsrc)
        return PK.P(self.zeval,self.k)

    def get_alphas(self,k_cents):
        #return np.ones(len(k_cents))
        return np.random.uniform(low=0.0,high=0.5,size=(len(k_cents),))

