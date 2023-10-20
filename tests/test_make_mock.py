import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import re

pkcl = pk.Cl_kk_supp()
zsrc_arr = np.array([0.2,0.4,0.6,0.8,1.0])
for z in zsrc_arr:
    pkcl.zsrc = z
    cents,data_binned,cinv = pkcl.make_mock_data()
    cl_arr = np.vstack((cents,data_binned))
    fn_data = "../mock_data/mock_cl_"+str(z)+".txt"
    fn_cinv = re.sub("cl","cinv",fn_data)
    np.savetxt(fn_data,cl_arr)
    np.savetxt(fn_cinv,cinv)
