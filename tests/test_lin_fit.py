import sys
sys.path.append("../")
import pk_func_fit_linear as pk

pkcl = pk.Cl_kk_supp()
X,chisquare = pkcl.eval()
print(X)
print(chisquare)


