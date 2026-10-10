# Monte-Carlo check of eq (7a): SMF coupling with tip-tilt removal, pupil-plane overlap integral
# (method: Fidler & Wallner 2008 §3; Dikmelik & Davidson 2005). Kolmogorov FFT screens + subharmonics.
import numpy as np, sys
rng=np.random.default_rng(1)
N=256; Lscr=4.0          # screen side in units of D
dx=Lscr/N
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); R=np.hypot(X,Y)
P=(R<=0.5).astype(float)  # pupil D=1
def screen(r0):
    fx=np.fft.fftfreq(N,dx); FX,FY=np.meshgrid(fx,fx); f=np.hypot(FX,FY); f[0,0]=1
    psd=0.023*r0**(-5/3)*f**(-11/3); psd[0,0]=0
    c=(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))*np.sqrt(psd)/(Lscr)
    ph=np.real(np.fft.ifft2(c))*N*N
    # Lane subharmonics (3 levels)
    for p in range(1,4):
        df=1/(3**p*Lscr)
        for i in (-1,0,1):
            for j in (-1,0,1):
                if i==0 and j==0: continue
                fxx,fyy=i*df,j*df; ff=np.hypot(fxx,fyy)
                a=(rng.normal()+1j*rng.normal())*np.sqrt(0.023*r0**(-5/3)*ff**(-11/3))*df
                ph+=np.real(a*np.exp(2j*np.pi*(fxx*X+fyy*Y)))
    return ph
m=P>0; xs,ys=X[m],Y[m]; A=np.c_[np.ones(xs.size),xs,ys]
def remove_tt(ph):
    c,*_=np.linalg.lstsq(A,ph[m],rcond=None); return ph[m]-A@c
def eta(phm,w):
    F=np.exp(-(xs**2+ys**2)/w**2)
    num=abs(np.sum(np.exp(1j*phm)*F))**2
    return num/(xs.size*np.sum(np.exp(-2*R**2/w**2)))
ws=np.linspace(0.2,0.6,81); e0=[eta(np.zeros(xs.size),w) for w in ws]
w=ws[int(np.argmax(e0))]; E0=max(e0)
print(f"eta0={E0:.4f} (theory 0.8145), w/a={w/0.5:.3f}")
nrep=int(sys.argv[1]) if len(sys.argv)>1 else 200
def S7a(d):
    s2=0.134*d**(5/3); return np.exp(-s2)+(1-np.exp(-s2))*(1+d**(5/3))**(-6/5)
print("D/r0  sim_TT(dB)  eq7a(dB)  diff  | sim_noTT(dB)  Fidler6(dB)  maréchal_core(dB)")
for d in [1,2,3,4,6,8,10,12]:
    r0=1/d; tt=[];nt=[]
    for k in range(nrep):
        ph=screen(r0); phm=ph[m]
        nt.append(eta(phm-phm.mean(),w)); tt.append(eta(remove_tt(ph),w))
    a=10*np.log10(np.mean(tt)/E0); b=10*np.log10(S7a(d)); c=10*np.log10(np.mean(nt)/E0)
    print(f"{d:4.0f}  {a:9.2f}  {b:8.2f}  {b-a:+5.2f} | {c:10.2f}  {10*np.log10((1+d**(5/3))**(-6/5)):9.2f}  {-0.134*d**(5/3)*4.343:9.2f}",flush=True)
