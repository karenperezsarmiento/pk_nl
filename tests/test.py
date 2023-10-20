import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import matplotlib.pyplot as plt

zsrc_arr = [0.2,0.4,0.6,0.8,1.0]
C_kk = {}
fig = plt.figure()
cl = pk.Cl_kk_supp()
for i in range(len(zsrc_arr)):
    cl.zsrc = zsrc_arr[i]
    C_kk[zsrc_arr[i]] = cl.get_Cl_kk()
    lab = "Src at z="+str(zsrc_arr[i])
    plt.loglog(cl.ells,cl.ells*C_kk[zsrc_arr[i]],label=lab)

cl.get_theo()
plt.loglog(cl.d["cents"],cl.d["cents"]*cl.d["data_binned"],label="Data binned")
plt.loglog(cl.d["cents"],cl.d["cents"]*cl.d["theory_binned"],label="Theory binned")
plt.legend()
plt.savefig("../plots/cl_kk_zsrc.png")
plt.close(fig)

