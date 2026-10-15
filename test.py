import time
import numpy as np
from scipy.special import ellipe, beta as Beta
import matplotlib
import matplotlib.pyplot as plt

a, b = 5.0, 2.0
A_exact = np.pi * a * b
m = 1 - b**2 / a**2                      # parameter m = e^2 for scipy
C_exact = 4 * a * ellipe(m)              # perimeter = 4 a E(e^2)
rng = np.random.default_rng(12345)

# ---------------------------------------------------------------- AREA
# every estimator returns (estimate, standard error from the sample itself)
def _mean_se(w, scale):
    return scale * w.mean(), scale * w.std(ddof=1) / np.sqrt(w.size)

def area_big_box(N, L=10.0):
    """naive: uniform over a box much bigger than the ellipse ('whole space')"""
    x = rng.uniform(-L, L, N); y = rng.uniform(-L, L, N)
    hit = (x/a)**2 + (y/b)**2 <= 1
    return _mean_se(hit.astype(float), (2*L)**2)

def area_tight_box(N):
    """uniform over the bounding box [-a,a]x[-b,b]"""
    x = rng.uniform(-a, a, N); y = rng.uniform(-b, b, N)
    hit = (x/a)**2 + (y/b)**2 <= 1
    return _mean_se(hit.astype(float), 4*a*b)

ALPHA = 1.25   # proposal shape for the designed estimator
def area_designed(N):
    """integrate y analytically (chord length), importance-sample x with
       q(x) ∝ (1-x²/a²)^(α-1), α=1.25, a proposal that mimics the chord length"""
    t = rng.beta(ALPHA, ALPHA, N); u = 2*t - 1; x = a*u
    q = (1 - u**2)**(ALPHA-1) / (a * 2**(2*ALPHA-1) * Beta(ALPHA, ALPHA)) 
    chord = 2*b*np.sqrt(1 - u**2)
    w = chord / q
    return w.mean(), w.std(ddof=1)/np.sqrt(N)

# ---------------------------------------------------------- CIRCUMFERENCE
f = lambda t: np.sqrt((a*np.sin(t))**2 + (b*np.cos(t))**2)   # |dr/dt|

def circ_uniform(N):
    """naive: t uniform on [0, 2π)"""
    t = rng.uniform(0, 2*np.pi, N)
    return _mean_se(f(t), 2*np.pi)

KAPPA = 0.34   # tuned below
def circ_designed(N, kappa=None):
    """importance sampling: q(t) = (1 - κ cos 2t)/(2π), largest where the
       ellipse is 'fastest' (near t = ±π/2). Invert the CDF by Newton."""
    k = KAPPA if kappa is None else kappa
    target = 2*np.pi*rng.uniform(size=N)
    t = target.copy()
    for _ in range(8):                         # solve t - (κ/2) sin 2t = target
        t -= (t - 0.5*k*np.sin(2*t) - target) / (1 - k*np.cos(2*t))
    q = (1 - k*np.cos(2*t)) / (2*np.pi)
    return _mean_se(f(t)/q, 1.0)

# tune kappa by exact variance (quadrature), not by sampling
tt = np.linspace(0, 2*np.pi, 400001)[:-1]
def circ_var(k):
    q = (1 - k*np.cos(2*tt))/(2*np.pi)
    return np.mean(f(tt)**2/q)*2*np.pi - C_exact**2     # E_q[(f/q)^2]-C^2
ks = np.linspace(0, 0.95, 191)
KAPPA = ks[np.argmin([circ_var(k) for k in ks])]
sigma1 = {"uniform t": np.sqrt(circ_var(0.0)), "designed": np.sqrt(circ_var(KAPPA))}

# ------------------------------------------------------- scaling study
Ns = np.array([10**2, 10**3, 10**4, 10**5, 10**6][:4])   # 1e2..1e5
Ns = np.unique(np.round(10**np.arange(2, 5.01, 0.5)).astype(int))
R = 200
methods = {
    "area: big box (L=10)": (area_big_box, A_exact),
    "area: tight box":      (area_tight_box, A_exact),
    "area: designed":       (area_designed, A_exact),
    "circ: uniform t":      (circ_uniform, C_exact),
    "circ: designed":       (circ_designed, C_exact),
}
res = {k: {"rep_std": [], "mean_se": [], "bias": []} for k in methods}
for k, (fn, exact) in methods.items():
    for N in Ns:
        out = np.array([fn(N) for _ in range(R)])
        res[k]["rep_std"].append(out[:, 0].std(ddof=1))
        res[k]["mean_se"].append(out[:, 1].mean())
        res[k]["bias"].append(out[:, 0].mean() - exact)

print(f"exact area = {A_exact:.6f}   exact circumference = {C_exact:.6f}   kappa* = {KAPPA:.3f}")
print("\nfitted slope of log(sigma) vs log(N)  (expect -0.5):")
slopes = {}
for k in methods:
    s = np.polyfit(np.log(Ns), np.log(res[k]["rep_std"]), 1)[0]
    slopes[k] = s
    print(f"  {k:24s} {s:+.3f}")

# ---------------------------------------------------- single big run
N0 = 10**6
print(f"\nsingle runs, N = {N0:.0e}")
timing = {}
for k, (fn, exact) in methods.items():
    t0 = time.perf_counter(); est, se = fn(N0); dt = time.perf_counter() - t0
    timing[k] = dt
    print(f"  {k:24s} est={est:10.5f}  se={se:.5f}  pull={(est-exact)/se:+.2f}  "
          f"exact={exact:.5f}  t={dt*1e3:6.1f} ms")

# --------------------------------------------------------- plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
(ax_area, ax_circ), (ax_pa, ax_pc) = axes

# ---- top row: scaling study (points = replica std, lines = mean single-run SE)
colors = {"area: big box (L=10)": "C3", "area: tight box": "C1", "area: designed": "C2",
          "circ: uniform t": "C3",       "circ: designed": "C2"}
for k in methods:
    ax = ax_area if k.startswith("area") else ax_circ
    c = colors[k]
    ax.loglog(Ns, res[k]["rep_std"], "o", color=c, label=f"{k} (slope {slopes[k]:+.2f})")
    ax.loglog(Ns, res[k]["mean_se"], "-", color=c, alpha=0.7)

for ax, title, keys in [(ax_area, "Area", [k for k in methods if k.startswith("area")]),
                        (ax_circ, "Circumference", [k for k in methods if k.startswith("circ")])]:
    # reference N^-1/2 line anchored at the first point of the first method
    y0 = res[keys[0]]["rep_std"][0]
    ax.loglog(Ns, y0 * (Ns / Ns[0])**-0.5, "k--", lw=1, label=r"$N^{-1/2}$")
    ax.set_title(title)
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("standard error")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

# ---- bottom row: what the proposals look like
# area: compare sampling densities in x with the (normalised) integrand = chord length
x = np.linspace(-a, a, 1000)
u = x / a
chord_density = 2*b*np.sqrt(1 - u**2) / A_exact              # ideal density, integrates to 1
q_des = (1 - u**2)**(ALPHA-1) / (a * 2**(2*ALPHA-1) * Beta(ALPHA, ALPHA))
ax_pa.plot(x, chord_density, "k", lw=2, label="ideal ∝ chord length")
ax_pa.plot(x, q_des, "C2", label=f"designed Beta({ALPHA},{ALPHA})")
ax_pa.plot(x, np.full_like(x, 1/(2*a)), "C1--", label="tight box (uniform)")
ax_pa.set_title("Area: proposal density in x")
ax_pa.set_xlabel("x"); ax_pa.legend(fontsize=8); ax_pa.grid(alpha=0.3)

# circumference: compare sampling densities in t with the normalised integrand f(t)
t = np.linspace(0, 2*np.pi, 1000)
ax_pc.plot(t, f(t) / C_exact, "k", lw=2, label="ideal ∝ f(t)")
ax_pc.plot(t, (1 - KAPPA*np.cos(2*t)) / (2*np.pi), "C2", label=f"designed (κ={KAPPA:.2f})")
ax_pc.plot(t, np.full_like(t, 1/(2*np.pi)), "C3--", label="uniform t")
ax_pc.set_title("Circumference: proposal density in t")
ax_pc.set_xlabel("t"); ax_pc.legend(fontsize=8); ax_pc.grid(alpha=0.3)

fig.suptitle("Top: points = std over 200 replicas, lines = mean single-run standard error.  "
             "Bottom: proposals vs. the ideal density.", fontsize=10)
fig.tight_layout()
plt.show()      # one window, four panels