import sys
sys.path.append("../")
import pk_func as pk
import numpy as np
import matplotlib.pyplot as plt


cents,data = np.loadtxt("../mock_data/mock_cl_0.4.txt")
cents,sup = np.loadtxt("../mock_data/mock_sup_cl_0.4.txt")

fig = plt.figure()
plt.loglog(cents,data,label="reg data")
plt.loglog(cents,sup,label="sup data")
plt.legend()
plt.savefig("../plots/reg_vs_sup_0.4.png")
plt.close(fig)
