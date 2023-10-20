import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import matplotlib.pyplot as plt

pkcl = pk.Cl_kk_supp()
pkcl.zsrc = 0.4
cents,data_binned,cinv = pkcl.make_mock_data(sup=True)
cl_arr = np.vstack((cents,data_binned))
np.savetxt("../mock_data/mock_sup_cl_0.4.txt",cl_arr)
np.savetxt("../mock_data/mock_sup_cinv_0.4.txt",cinv)
