import sys
import numpy as np
sys.path.append("../")
import pandas as pd
import pk_func_fit_linear as pk
import matplotlib.pyplot as plt
p = pk.Cl_kk_supp()
p.load_data()
alphas = np.ones(3)
logp = p.logp(alphas)

fig = plt.figure()
plt.loglog(p.d["cents"],p.d["theory_binned"],label="theory")
plt.loglog(p.d["cents"],p.d["data_binned"],label="data")
plt.legend()
plt.savefig("act_data_theo.png")
plt.close(fig)

print(logp)



print(p.d["cents"].shape)
print(p.d["data_binned"].shape)
print(p.d["theory_binned"].shape)
