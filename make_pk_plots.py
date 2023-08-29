import pk_alpha
import matplotlib.pyplot as plt

zevals = [0.2,0.4,0.6,0.8,1.0]
for z in zevals:
    pk_alpha.Generate_Pk_Alpha.zeval = z
    pk = pk_alpha.Generate_Pk_Alpha().Pk_sup
    k_cents = pk_alpha.Generate_Pk_Alpha().k_cents
    plt.loglog(k_cents,pk,label=str(z))
    plt.legend()
    plt.savefig("pk_plot.png")
