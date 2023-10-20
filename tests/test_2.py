import pk_func as pk
import numpy as np
#from pk_func import Cl_kk_supp as cl
import matplotlib.pyplot as plt

pkcl = pk.Cl_kk_supp()
pkcl.zsrc = 0.4
alphas = np.arange(20)/40
cents,data_binned,cinv = pkcl.make_mock_data(alphas)
cl_arr = np.vstack((cents,data_binned))
np.savetxt("../mock_data/mock_sup_cl_0.4.txt",cl_arr)
np.savetxt("../mock_data/mock_sup_cinv0.4.txt",cinv)