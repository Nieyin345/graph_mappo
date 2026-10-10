# Dikmelik & Davidson 2005 (Appl. Opt. 44, 4946) mean SMF coupling, no tracking:
# eta = 8 a^2 ∫∫ exp[-(a^2+A)(x1^2+x2^2)] I0(2 A x1 x2) x1 x2 dx1 dx2, A=(D/2)^2/rho0^2, rho0=r0/2.1
import numpy as np
from scipy.special import i0e
from scipy import integrate
a=1.12
def eta(d):
    A=(0.5*d*2.1)**2
    f=lambda x2,x1: np.exp(-(a*a+A)*(x1*x1+x2*x2)+2*A*x1*x2)*i0e(2*A*x1*x2)*x1*x2
    v,_=integrate.dblquad(f,0,1,0,1); return 8*a*a*v
e0=eta(1e-9)
for d in [1,2,3,4,6,8,10,12]: print(d, round(10*np.log10(eta(d)/e0),2))
print('eta0',round(e0,4))
