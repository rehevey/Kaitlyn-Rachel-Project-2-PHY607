from Particle_Source_Class import Particle_Source
#utilizes CollisionModel class
from Collision-Class-Benchmark import CollisionModel
import numpy as np

#----Let's Validate Kepler's Third Law----

#Kepler's 3rd Law: The square of the orbital period is proportional to the cube of the semi-major axis

#Our single body model that will follow the orbit
def single_body(a, ecc=0.0):
    v = np.sqrt(G * Msun / a) * np.sqrt(1 - ecc)
    # a = starting distance. the true semi-major axis is a / (1 + ecc)
    return (np.array([[a, 0, 0]]), np.array([[0, 0, -v]]), np.array([1e-12]), np.array([1e-9]))


#period = time it takes for 1 complete orbit
def measure_period(a, ecc=0.0, t_max=None, dt=0.002, **kw):
    pos, vel, m, rad = single_body(a, ecc)
    mdl = CollisionModel(pos, vel, m, rad, m0=1e-12, rad0=1e-9, **kw)
    a_true = a / (1 + ecc)
    t_max = t_max or 4 * a_true**1.5
    ts, ang = [0.0], [np.arctan2(mdl.pos[0, 2], mdl.pos[0, 0])]
    while mdl.t < t_max:
        mdl.advance(dt)
        ts.append(mdl.t)
        ang.append(np.arctan2(mdl.pos[0, 2], mdl.pos[0, 0]))
    th = np.abs(np.unwrap(ang) - ang[0])
    ts = np.array(ts)
    n = int(th[-1] // (2 * np.pi))
    # time of each full revolution by linear interpolation
    crossings = [np.interp(2 * np.pi * k, th, ts) for k in range(1, n + 1)]
    return np.diff([0] + crossings).mean(), a_true

for a in (0.5, 1, 2, 4):
    T, a_true = measure_period(a)
    print(a, T, a_true**1.5, abs(T - a_true**1.5) / a_true**1.5)

a_vals = np.array([0.5, 1, 2, 4])
T_vals = np.array([measure_period(a)[0] for a in a_vals])

slope, _ = np.polyfit(np.log(a_vals), np.log(T_vals), 1)
print("log T vs log a slope:", slope, "(expected 1.5)")

plt.loglog(a_vals**3, T_vals**2, "o", label="measured")
plt.loglog(a_vals**3, a_vals**3, "--", label="T² = a³")
plt.xlabel("a³ (AU³)"); plt.ylabel("T² (yr²)")
plt.legend()
plt.title("Kepler's Third Law")
plt.show()
