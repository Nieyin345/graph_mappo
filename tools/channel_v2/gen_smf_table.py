"""Tabulate the mean single-mode-fibre coupling efficiency of a tip-tilt-corrected receiver,
normalised to the turbulence-free value, eta_phi/eta0, versus D/r0.

Method (all from the literature, no fitted formula):
  * coupling efficiency = pupil-plane overlap integral of the incident field with the
    back-propagated fibre mode  [Fidler & Wallner 2008, eq. (1)]; Gaussian fibre mode with the
    optimum coupling parameter, which gives eta0 = 0.8145 [Karakosta 2025, eq. (C11)];
  * Kolmogorov phase screens, FFT method with subharmonic compensation
    [Lane, Glindemann & Dainty 1992, Waves Random Media 2, 209];
  * tip-tilt correction = least-squares removal of piston, tip and tilt over the pupil; the residual
    variance is checked against Noll's 0.134 (D/r0)^{5/3} [Noll 1976, Table IV].
Phase scales as r0^{-5/6}, so every unit screen (D/r0 = 1) is reused for all D/r0 values.
Usage: python gen_smf_table.py N_screens N_grid  -> prints a Python table.
"""
import sys
import numpy as np

def run(nscr=1000, N=512, Lscr=4.0, seed=12345, dlist=None):
    rng = np.random.default_rng(seed)
    dx = Lscr / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    m = R <= 0.5                      # pupil, D = 1
    xs, ys = X[m], Y[m]
    A = np.c_[np.ones(xs.size), xs, ys]
    pinv = np.linalg.pinv(A)
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    f = np.hypot(FX, FY); f[0, 0] = 1.0
    amp = np.sqrt(0.023 * f ** (-11 / 3)) / Lscr; amp[0, 0] = 0.0   # r0 = 1 (D/r0 = 1)
    sub = []
    for p in range(1, 4):
        df = 1.0 / (3 ** p * Lscr)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i or j:
                    fxx, fyy = i * df, j * df
                    ff = np.hypot(fxx, fyy)
                    sub.append((np.sqrt(0.023 * ff ** (-11 / 3)) * df,
                                np.exp(2j * np.pi * (fxx * xs + fyy * ys))))
    # optimum Gaussian mode width (turbulence-free)
    def eta_of(phm, w):
        F = np.exp(-(xs ** 2 + ys ** 2) / w ** 2)
        return abs(np.sum(np.exp(1j * phm) * F)) ** 2 / (xs.size * np.sum(np.exp(-2 * R ** 2 / w ** 2)))
    ws = np.linspace(0.3, 0.6, 301)
    e0s = [eta_of(np.zeros(xs.size), w) for w in ws]
    w = ws[int(np.argmax(e0s))]; eta0 = max(e0s)
    F = np.exp(-(xs ** 2 + ys ** 2) / w ** 2)
    normF = xs.size * np.sum(np.exp(-2 * R ** 2 / w ** 2))
    if dlist is None:
        dlist = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16,
                          20, 24, 28, 32, 40, 48, 56, 64])
    s53 = dlist ** (5 / 6)
    acc = np.zeros(len(dlist)); acc2 = np.zeros(len(dlist)); var_tt = 0.0
    for k in range(nscr):
        c = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * amp
        ph = np.real(np.fft.ifft2(c))[m] * N * N
        for a, e in sub:
            ph = ph + np.real((rng.normal() + 1j * rng.normal()) * a * e)
        res = ph - A @ (pinv @ ph)                       # piston + tip/tilt removed, D/r0 = 1
        var_tt += np.var(res)
        for i, s in enumerate(s53):
            v = abs(np.sum(np.exp(1j * s * res) * F)) ** 2 / normF
            acc[i] += v; acc2[i] += v * v
    mean = acc / nscr
    sem = np.sqrt(np.maximum(acc2 / nscr - mean ** 2, 0) / nscr)
    return dlist, mean / eta0, sem / eta0, eta0, w / 0.5, var_tt / nscr / 0.134

if __name__ == "__main__":
    nscr = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    d, r, s, e0, wa, vr = run(nscr, N)
    print(f"# nscr={nscr} N={N} eta0={e0:.4f} w/a={wa:.3f} TT-residual var / Noll = {vr:.3f}")
    print("# D/r0, eta_phi/eta0, dB, rel.SEM")
    for a, b, c in zip(d, r, s):
        print(f"{a:6.2f}, {b:.6e}, {10*np.log10(b):7.2f}, {c/b:.3f}")
