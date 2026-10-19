from vpython import *
import numpy as np

scene = canvas(title="N-body", width=900, height=700, background=color.black)
scene.autoscale = False
scene.range = 1.8

# ---------- parameters ----------
G = 4.0 * np.pi**2          # AU, years, solar masses
Msun = 1.0
m0 = 1e-7                   # planetesimal mass (solar masses)
rad0 = 0.015                # enhanced collision radius (AU)
r_in, r_out = 0.8, 1.5      # annulus (AU)
N = 100
e_rest = 0.5                # coefficient of restitution for bounces

t = 0.0
dt = 0.7**1.5 / 150
t_end = 100                 # simulated years
steps_per_frame = 10        # raise this to run faster

rng = np.random.default_rng(42)

# ---------- initial conditions (numpy arrays) ----------
r = rng.uniform(r_in, r_out, N)
th = rng.uniform(0, 2*np.pi, N)
pos = np.column_stack((r*np.cos(th), 0.002*r*rng.normal(size=N), r*np.sin(th)))
v_circ = np.sqrt(G*Msun/r)
speed = v_circ * (1 + 0.005*rng.normal(size=N))
vel = np.column_stack((np.sin(th), np.zeros(N), -np.cos(th))) * speed[:, None]
vel[:, 1] += 0.003 * v_circ * rng.normal(size=N)

m = np.full(N, m0)
radius = np.full(N, rad0)
alive = np.ones(N, dtype=bool)

# ---------- graphics (display only) ----------
sun = sphere(pos=vector(0, 0, 0), radius=0.05, color=color.yellow, emissive=True)
spheres = [sphere(pos=vector(*pos[k]), radius=rad0, make_trail=False) for k in range(N)]

# ---------- statistics ----------
collisions = merges = bounces = 0
touching = set()
stats = wtext(text="")

# ---------- collision helpers ----------
def merge_probability(a, b):
    v_rel = np.linalg.norm(vel[b] - vel[a])
    v_esc = np.sqrt(2*G*(m[a] + m[b]) / (radius[a] + radius[b]))
    return 1.0 / (1.0 + (v_rel/v_esc)**2)

def do_merge(a, b):
    m_new = m[a] + m[b]
    pos[a] = (m[a]*pos[a] + m[b]*pos[b]) / m_new
    vel[a] = (m[a]*vel[a] + m[b]*vel[b]) / m_new     # momentum conserved
    m[a] = m_new
    radius[a] = rad0 * (m_new/m0)**(1/3)
    alive[b] = False
    spheres[a].radius = radius[a]
    spheres[b].visible = False

def do_bounce(a, b):
    n = pos[b] - pos[a]
    n = n / np.linalg.norm(n)
    vn = np.dot(vel[b] - vel[a], n)
    if vn < 0:                                       # only if still approaching
        J = (1 + e_rest) * vn * m[a] * m[b] / (m[a] + m[b])
        vel[a] += J*n / m[a]
        vel[b] -= J*n / m[b]

# ---------- one physics step ----------
def step():
    global collisions, merges, bounces, touching

    idx = np.flatnonzero(alive)
    P = pos[idx]
    d = P[None, :, :] - P[:, None, :]                # d[i,j] = P[j] - P[i]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    contact = dist < (radius[idx][:, None] + radius[idx][None, :])

    # mutual gravity (skipped for pairs in contact, as before)
    safe = np.where(contact, np.inf, dist)
    acc = (G * m[idx][None, :, None] * d / safe[:, :, None]**3).sum(axis=1)
    vel[idx] += acc * dt

    # collisions: only loop over the few pairs actually touching
    ii, jj = np.nonzero(np.triu(contact, 1))
    current = set()
    for a, b in zip(idx[ii], idx[jj]):
        a, b = int(a), int(b)
        current.add((a, b))
        if not (alive[a] and alive[b]):
            continue
        if (a, b) not in touching:                   # new collision
            collisions += 1
            if rng.random() < merge_probability(a, b):
                do_merge(a, b)
                merges += 1
            else:
                do_bounce(a, b)
                bounces += 1
                touching.add((a, b))
        else:                                        # still overlapping after a bounce
            do_bounce(a, b)
    touching &= current                              # forget pairs that separated

    # sun gravity and move
    al = alive
    r3 = np.linalg.norm(pos[al], axis=1)**3
    vel[al] -= G * Msun * pos[al] / r3[:, None] * dt
    pos[al] += vel[al] * dt

# ---------- main loop ----------
while t < t_end:
    rate(60)
    for _ in range(steps_per_frame):
        step()
        t += dt

    for k in np.flatnonzero(alive):                  # update drawing once per frame
        spheres[k].pos = vector(*pos[k])

    stats.text = (f"t = {t:.1f} | collisions: {collisions} | merges: {merges} | "
                  f"bounces: {bounces} | protoplanets left: {int(alive.sum())}")

print('Simulation finished')
print(f"Collisions: {collisions}")
print(f"Merges: {merges}")
print(f"Bounces: {bounces}")
print(f"Protoplanets left: {int(alive.sum())}")
while True:
    rate(10)