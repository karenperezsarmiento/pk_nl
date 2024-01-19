import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    temp_string = np.array2string(a, formatter={'float_kind':lambda x: "{:.2e}".format(x)})
    lines = temp_string.replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2

a = np.ones((3,3))

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(t, s)
tb = bmatrix(a)
ax.text(0.05,0.95,tb)
#ax.set_xlabel(r'\textbf{time (s)}')
#ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')

plt.savefig("latex.png")
