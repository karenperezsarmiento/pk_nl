import numpy as np
import os
import pyfisher as pf
from scipy.interpolate import interp1d
import camb
from camb import model

H0 = 67.5
ombh2 = 0.022
omch2 = 0.122
ns = 0.965
k_min = 1e-3
k_max = 10
zsrc = 620.7057
zeval = 0.2

def get_pure_Pk(k):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=ns)
    results = camb.get_background(pars)
    PK = camb.get_matter_power_interpolator(pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=k_max,var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zsrc)
    return PK.P(zeval,k)

def get_Pk_suppressed(ks,alphas):
    Pk = get_pure_Pk(ks)
    Pk_sup = alphas*Pk
    return Pk_sup,Pk

def get_chi(ks,alphas):
    Pk_sup,Pk = get_Pk_suppressed(ks,alphas)
    chi_sq = np.sum((Pk_sup - Pk)**2/Pk)
    return chi_sq
