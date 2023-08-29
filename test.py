import pk_func as pk
import numpy as np

ks = np.geomspace(1e-3,10,30)
alphas = np.random.random_sample(len(ks)) - 0.5

chi_sq = pk.get_chi(ks,alphas)
print(chi_sq)
