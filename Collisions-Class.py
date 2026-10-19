from vpython import * 
import numpy as np
import random as rng

scene = canvas(title="N-body", width=900, height=700, background=color.black)
scene.autoscale = False
scene.range = 1.8

G = 4.0 * np.pi**2          # AU, years, solar masses
Msun = 1.0
m0 = 1e-7                   # each planetesimal, in solar masses (~0.03 Earth masses)
# rad = 0.004                 # enhanced collision radius in AU (toy value, bigger than physical)
rad = 0.015
r_in, r_out = 0.8, 1.5      # annulus in AU

t = 0   # time value
dt = 0.7**1.5 / 150    # time step 

N = 100 

protoplanets = [] 

collisions = 0
merges = 0
bounces = 0
touching = set()      # pairs currently in contact
stats = wtext(text="")

for _ in range(N):
    r = rng.uniform(r_in, r_out)
    th = rng.uniform(0, 2*np.pi)
    pos = vector(r*np.cos(th), 0.002*r*rng.gauss(0, 1), r*np.sin(th))        # changed: 0.01 -> 0.002
    rhat = norm(vector(pos.x, 0, pos.z))
    v_circ = np.sqrt(G*Msun/r)
    vel = cross(vector(0,1,0), rhat) * v_circ * (1 + 0.005*rng.gauss(0, 1))  # changed: 0.03 -> 0.005
    vel.y += 0.003*v_circ*rng.gauss(0, 1)                                    # changed: 0.02 -> 0.003
    b = sphere(pos=pos, radius=rad, make_trail=False)
    b.m = m0
    b.p = m0*vel
    b.alive = True
    protoplanets.append(b)

class CollisionModel:

    # Collisions are either a merger (perfect inelastic) or a bounce (inelastic but not perfect). 
    # The choice is a random draw with the merger probability that adjusts with the impact speed. 

    # ----- Random Number Generator -----

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)


    def merge_probability(self, i, j):
        a, b = protoplanets[i], protoplanets[j]
        v_rel = mag(b.p/b.m - a.p/a.m)
        v_esc = sqrt(2*G*(a.m + b.m)/(a.radius + b.radius))
        return 1.0/(1.0 + (v_rel/v_esc)**2)

    
    def outcome(self, i, j):
        if self.rng.random() < self.merge_probability(i, j):
            self._merge(i, j)
            return True
        self._bounce(i, j)
        return False

 
    def _merge(self, i, j):
        a, b = protoplanets[i], protoplanets[j]
        m_new = a.m + b.m
        a.pos = (a.m*a.pos + b.m*b.pos)/m_new     # center of mass
        a.p = a.p + b.p                           # momentum is conserved
        a.m = m_new
        a.radius = rad*(m_new/m0)**(1/3)          # grow with mass
        b.alive = False
        b.visible = False


    def _bounce(self, i, j, e=0.5):          # e = coefficient of restitution
        a, b = protoplanets[i], protoplanets[j]
        n = norm(b.pos - a.pos)
        vn = dot(b.p/b.m - a.p/a.m, n)       # relative speed along line of centers
        if vn < 0:                           # only act if still approaching
            J = (1 + e) * vn * a.m * b.m / (a.m + b.m)
            a.p = a.p + J*n
            b.p = b.p - J*n


def gravity(i, j):
        a, b = protoplanets[i], protoplanets[j]
        r = b.pos - a.pos
        F = G*a.m*b.m*norm(r)/mag(r)**2
        a.p = a.p + F*dt
        b.p = b.p - F*dt
 

model = CollisionModel(seed=42)

sun = sphere(pos=vector(0,0,0), radius=0.05, color=color.yellow, emissive=True)
sun.m = Msun

def sun_gravity(b):
    r = sun.pos - b.pos
    F = G * sun.m * b.m * r / mag(r)**3
    b.p = b.p + F*dt

while t < 10:
    rate(60)

    for _ in range(5):
        live = [k for k, b in enumerate(protoplanets) if b.alive]

        for x in range(len(live)):
            for y in range(x+1, len(live)):
                i, j = live[x], live[y]
                if not (protoplanets[i].alive and protoplanets[j].alive):
                    continue
                d = mag(protoplanets[j].pos - protoplanets[i].pos)
                if d < protoplanets[i].radius + protoplanets[j].radius:
                    if (i, j) not in touching:          # new collision
                        collisions += 1
                        if model.outcome(i, j):
                            merges += 1
                        else:
                            bounces += 1
                            touching.add((i, j))
                    else:                               # still overlapping after a bounce
                        model._bounce(i, j)
                else:
                    touching.discard((i, j))
                    gravity(i, j)

        for b in protoplanets:
            if b.alive:
                sun_gravity(b)
                b.pos = b.pos + b.p/b.m*dt

        t += dt
    remaining = sum(1 for b in protoplanets if b.alive)
    stats.text = (f"t = {t:.1f} | collisions: {collisions} | merges: {merges} | "
                  f"bounces: {bounces} | protoplanets left: {remaining}")


print('Simulation finished')
print(f"Collisions: {collisions}")
print(f"Merges: {merges}")
print(f"Bounces: {bounces}")
print(f"Protoplanets left: {sum(1 for b in protoplanets if b.alive)}")
while True:
    rate(10)
