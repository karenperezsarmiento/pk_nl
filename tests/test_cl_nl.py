import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import matplotlib.pyplot as plt


cents,data = np.loadtxt("../mock_data/mock_cl_0.5.txt")
nl = np.loadtxt("../mock_data/mock_noise_0.5.txt")

fig = plt.figure()
plt.loglog(cents,data,label="C_L")
plt.loglog(np.arange(len(nl)),nl,label="N_L")
plt.legend()
plt.savefig("../plots/cl_nl_0.5.png")
plt.close(fig)
