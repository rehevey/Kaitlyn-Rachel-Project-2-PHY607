import numpy as np
from scipy.integrate import solve_ivp

# Units: AU, years, solar masses

G = 4.0 * np.pi**2
Msun = 1.0


# =====================================================================
#  Physics model: fuck VPython, I want a divorce and I'm taking the kids 
# =====================================================================

class CollisionModel:

    # State is held in numpy arrays (one row per body):
        #pos, vel : (N, 3)   m, radius, alive : (N,)

    #The ODE state passed to solve_ivp is y = [x1,y1,z1, x2,..., vx1,vy1,vz1, vx2,...]
    #and dy/dt = [v, a(r)].  Dead (oen of the mergies) bodies are dropped from y each call.
    
    def __init__(self, pos, vel, m, radius, m0, rad0, seed=None,e_rest=0.5, eps=None,rtol=1e-9, atol=1e-12,contact_tol=1e-6, separation=1e-4):
        
        self.pos = np.array(pos, dtype=float)       # position
        self.vel = np.array(vel, dtype=float)       # velocity 
        self.m = np.array(m, dtype=float)           
        self.radius = np.array(radius, dtype=float) # radius
        self.alive = np.ones(len(self.m), dtype=bool) # starts all bodies as alive ("on")
        self.N = len(self.m)                         # holding onto the number of bodies currently in play 

        self.m0 = m0                          # reference mass for the radius-growth law
        self.rad0 = rad0                      # radius of a mass=m0 body
        self.e_rest = e_rest                  # coefficient of restitution for bounces
        self.eps = 0.5 * rad0 if eps is None else eps   # gravitational softening (AU)
        self.rtol = rtol
        self.atol = atol
        self.contact_tol = contact_tol        # relative tolerance when testing overlap
        self.separation = separation          # relative gap enforced after a bounce

        self.rng = np.random.default_rng(seed)
        self.t = 0.0                            # initial time set to 0
        self.collisions = 0                     # counters for number of collisions & how many are merges or bounces 
        self.merges = 0
        self.bounces = 0

        self.collision_log = []                 # one (t, v_rel/v_esc, merged?) per collision

    # ---------- ODE system ----------
    @staticmethod       #doesn't need access to the instance, doesnn't need self 
    def unpack(y, n):
        return y[:3*n].reshape(n, 3), y[3*n:].reshape(n, 3)     #uses numpy's reshape to change the array shape -- unsmush the info from a flat 1D array for solve_ivp

    def rhs(self, t, y, mass):
        n = len(mass)
        P, V = self.unpack(y, n)

        # mutual gravity (softened; d[i,j] = P[j] - P[i], d[i,i] = 0 so no self-force)
        d = P[None, :, :] - P[:, None, :]
        s2 = np.einsum('ijk,ijk->ij', d, d) + self.eps**2
        acc = (G * mass[None, :, None] * d / s2[:, :, None]**1.5).sum(axis=1)

        # P[None, :, :]  -> shape (1, n, 3)   indexed as [0, j, k]  = P[j, k]
        # P[:, None, :]  -> shape (n, 1, 3)   indexed as [i, 0, k]  = P[i, k]

        # Sun
        r3 = np.linalg.norm(P, axis=1)**3
        acc -= G * Msun * P / r3[:, None]

        return np.concatenate([V.ravel(), acc.ravel()])     # ravel flattens the array back to 1D so it's prepped for the solver

    def contact_event(self, t, y, mass, rad):

        #Smallest surface-to-surface gap over all pairs; crosses zero at first contact.
        
        n = len(mass)
        if n < 2:
            return 1.0
        P, _ = self.unpack(y, n)
        dist = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=2)
        gap = dist - (rad[:, None] + rad[None, :])
        np.fill_diagonal(gap, np.inf)
        return gap.min()

    def _make_event(self, rad):
    
        #solve_ivp passes the same args to events as to rhs, so close over the radii.

        def ev(t, y, mass):
            return self.contact_event(t, y, mass, rad)
        ev.terminal = True            # stop integrating at contact
        ev.direction = -1             # only trigger when the gap goes pos -> neg
        return ev

    # ---------- collision helpers ----------

    def vrel_over_vesc(self, a, b):
        v_rel = np.linalg.norm(self.vel[b] - self.vel[a])
        v_esc = np.sqrt(2*G*(self.m[a] + self.m[b]) / (self.radius[a] + self.radius[b]))
        return v_rel / v_esc

    def merge_probability(self, a, b):
        return 1.0 / (1.0 + self.vrel_over_vesc(a, b)**2)

    def do_merge(self, a, b):
        m, pos, vel = self.m, self.pos, self.vel
        m_new = m[a] + m[b]
        pos[a] = (m[a]*pos[a] + m[b]*pos[b]) / m_new
        vel[a] = (m[a]*vel[a] + m[b]*vel[b]) / m_new     # momentum conserved
        m[a] = m_new
        self.radius[a] = self.rad0 * (m_new/self.m0)**(1/3)
        self.alive[b] = False

    def do_bounce(self, a, b):
        m, pos, vel, radius = self.m, self.pos, self.vel, self.radius
        n = pos[b] - pos[a]
        dist = np.linalg.norm(n)
        n = n / dist
        vn = np.dot(vel[b] - vel[a], n)
        if vn < 0:                                       # only if still approaching
            J = (1 + self.e_rest) * vn * m[a] * m[b] / (m[a] + m[b])
            vel[a] += J*n / m[a]
            vel[b] -= J*n / m[b]

        # push the pair apart (about their centre of mass) so they are no longer overlapping; otherwise the contact event would start at a negative gap
        target = (radius[a] + radius[b]) * (1 + self.separation)
        shift = target - dist
        if shift > 0:
            pos[a] -= n * shift * m[b] / (m[a] + m[b])
            pos[b] += n * shift * m[a] / (m[a] + m[b])

    def find_overlap(self):
        #Return one overlapping pair (a, b) of alive bodies, or None.
        idx = np.flatnonzero(self.alive)
        if len(idx) < 2:
            return None
        P = self.pos[idx]
        dist = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        rr = self.radius[idx]
        contact = dist < (rr[:, None] + rr[None, :]) * (1 + self.contact_tol)
        ii, jj = np.nonzero(np.triu(contact, 1))
        if len(ii) == 0:
            return None
        return int(idx[ii[0]]), int(idx[jj[0]])

    def handle_collisions(self):
        #Resolve every overlapping pair (merge or bounce)
        for _ in range(4*self.N):                        # safety cap
            pair = self.find_overlap()
            if pair is None:
                return
            a, b = pair
            self.collisions += 1
            ratio = self.vrel_over_vesc(a, b)            # measure BEFORE merge/bounce changes velocities
            merged = self.rng.random() < 1.0 / (1.0 + ratio**2)
            self.collision_log.append((self.t, ratio, merged))
            if merged:
                self.do_merge(a, b)
                self.merges += 1
            else:
                self.do_bounce(a, b)
                self.bounces += 1

    # ---------- advance with solve_ivp ----------

    def advance(self, T):
        # Integrate all alive bodies forward by T years, stopping at each contact.
        t_left = T      #add another time variable because why not 
        for _ in range(10*self.N):                       # safety cap on restarts
            if t_left <= 1e-12:
                break
            idx = np.flatnonzero(self.alive)             # re-pack: n changes after merges
            n = len(idx)
            y0 = np.concatenate([self.pos[idx].ravel(), self.vel[idx].ravel()])
            ev = self._make_event(self.radius[idx].copy()) if n > 1 else None

            sol = solve_ivp(self.rhs, (0.0, t_left), y0, args=(self.m[idx],), method="DOP853", rtol=self.rtol, atol=self.atol, events=ev)
            if not sol.success:
                raise RuntimeError(sol.message)

            self.pos[idx], self.vel[idx] = self.unpack(sol.y[:, -1], n)
            dt_done = sol.t[-1]
            t_left -= dt_done
            self.t += dt_done

            if sol.status == 1:                          # stopped at a contact
                self.handle_collisions()

    # ---------- give me that data ----------
    @property
    def n_alive(self):
        return int(self.alive.sum())

    def summary(self):
        return (f"t = {self.t:.1f} | collisions: {self.collisions} | merges: {self.merges} | "
                f"bounces: {self.bounces} | protoplanets left: {self.n_alive}")


# =====================================================================
#  Initial conditions: swap this function for Kaitlyn's class.
#  All the model needs is pos (N,3), vel (N,3), m (N,), radius (N,).
# =====================================================================

def make_initial_conditions(N=100, m0=1e-7, rad0=0.015, r_in=0.8, r_out=1.5, seed=42):
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_in, r_out, N)
    th = rng.uniform(0, 2*np.pi, N)
    pos = np.column_stack((r*np.cos(th), 0.002*r*rng.normal(size=N), r*np.sin(th)))
    v_circ = np.sqrt(G*Msun/r)
    speed = v_circ * (1 + 0.005*rng.normal(size=N))
    vel = np.column_stack((np.sin(th), np.zeros(N), -np.cos(th))) * speed[:, None]
    vel[:, 1] += 0.003 * v_circ * rng.normal(size=N)
    m = np.full(N, m0)
    radius = np.full(N, rad0)
    return pos, vel, m, radius


# =====================================================================
#  Graphics (display only) -- matplotlib
# =====================================================================
 
def run_headless(model, t_end, frame_dt):
    #No graphics: just advance the model and print progress.
    next_report = 0.0
    while model.t < t_end:
        model.advance(frame_dt)
        if model.t >= next_report:
            print(model.summary())
            next_report += 5.0
 
def histogram_counts(model, bins):
    #Return (merge_counts, bounce_counts) for the logged v_rel/v_esc values. Out-of-range values land in the end bins.
    if not model.collision_log:
        z = np.zeros(len(bins) - 1)
        return z, z.copy()
    ratio = np.array([c[1] for c in model.collision_log])
    merged = np.array([c[2] for c in model.collision_log], dtype=bool)
    ratio = np.clip(ratio, bins[0], bins[-1] * (1 - 1e-12))   # out-of-range values go in the end bins
    return (np.histogram(ratio[merged], bins=bins)[0],
            np.histogram(ratio[~merged], bins=bins)[0])
 
 
def style_hist_axis(ax, xmin, xmax):
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.tick_params(colors="white")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"$v_{rel}/v_{esc}$ at collision", color="white")
    ax.set_ylabel("collisions", color="white")
    ax.set_title("Collision speeds", color="white")
 
 
def plot_histogram(model, xmin=0.1, xmax=100.0, nbins=40):
    #Static histogram of the finished run (use after headless mode).
    import matplotlib.pyplot as plt
    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    mc, bc = histogram_counts(model, bins)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("black")
    w = np.diff(bins)
    ax.bar(bins[:-1], mc, width=w, align="edge", color="limegreen", label="merge")
    ax.bar(bins[:-1], bc, width=w, align="edge", bottom=mc, color="tomato", label="bounce")
    style_hist_axis(ax, xmin, xmax)
    ax.legend(facecolor="black", labelcolor="white")
    plt.show()

 
def run_animation(model, t_end, frame_dt, view_range=1.8, save_path=None, steps_per_frame=1,hist_xmin=0.1, hist_xmax=100.0, hist_bins=40):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
 
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("N-body (SciPy DOP853 + collision events)", color="white")
 
    ax.scatter([0], [0], s=150, color="yellow", zorder=3)          # the Sun
    bodies = ax.scatter([], [], color="deepskyblue", zorder=2)
    stats = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white", va="top", fontsize=9)

    # histogram panel: merges (green) with bounces (red) stacked on top
    bins = np.logspace(np.log10(hist_xmin), np.log10(hist_xmax), hist_bins + 1)
    w = np.diff(bins)
    zeros = np.zeros(hist_bins)
    merge_bars = axh.bar(bins[:-1], zeros, width=w, align="edge", color="limegreen", label="merge")
    bounce_bars = axh.bar(bins[:-1], zeros, width=w, align="edge", bottom=zeros, color="tomato", label="bounce")
    style_hist_axis(axh, hist_xmin, hist_xmax)
    axh.legend(facecolor="black", labelcolor="white", loc="upper right")
 
    def marker_sizes(radii):
        # convert physical radius (AU) -> scatter area (points^2) so sizes track real size
        fig.canvas.draw_idle()
        width_pts = ax.get_window_extent().width * 72 / fig.dpi
        pts_per_au = width_pts / (2 * view_range)
        diam_pts = 2 * radii * pts_per_au
        return np.maximum(diam_pts, 2.0) ** 2

    def update_hist():
        mc, bc = histogram_counts(model, bins)
        for bar, h in zip(merge_bars, mc):
            bar.set_height(h)
        for bar, h, base in zip(bounce_bars, bc, mc):
            bar.set_height(h)
            bar.set_y(base)
        axh.set_ylim(0, max(5, (mc + bc).max() * 1.1))


    def update(_):
        for _ in range(steps_per_frame):
            model.advance(frame_dt)
        a = np.flatnonzero(model.alive)
        bodies.set_offsets(model.pos[a][:, [0, 2]])               # x vs z = orbital plane
        bodies.set_sizes(marker_sizes(model.radius[a]))
        stats.set_text(model.summary())
        update_hist()
        return bodies, stats

    def frames():
        while model.t < t_end:
            yield model.t
 
    fig.tight_layout()
    if save_path:
        n_frames = int(np.ceil(t_end / (frame_dt * steps_per_frame)))
        anim = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False)
        anim.save(save_path, fps=30, dpi=100)
        print(f"Saved {save_path}")
    else:
        anim = FuncAnimation(fig, update, frames=frames, interval=16, blit=False,cache_frame_data=False, repeat=False)
        plt.show()
    return anim
 
 
def main():
    # ---------- parameters ----------
    N = 100
    m0 = 1e-7                   # planetesimal mass (solar masses)
    rad0 = 0.015                # enhanced collision radius (AU)
    t_end = 100                 # simulated years
    frame_dt = 0.05             # years advanced per animation frame (solver subdivides)
 
    MODE = "animate"            # "animate" | "headless"
    SAVE_PATH = None            # e.g. "collisions.gif" to write a file instead of opening a window
 
    # ---------- model ----------
    pos, vel, m, radius = make_initial_conditions(N=N, m0=m0, rad0=rad0, seed=42)
    model = CollisionModel(pos, vel, m, radius, m0=m0, rad0=rad0, seed=42, e_rest=0.5, eps=0.5*rad0)
 
    # ---------- run ----------
    if MODE == "headless":
        run_headless(model, t_end, frame_dt)
        plot_histogram(model)
    else:
        run_animation(model, t_end, frame_dt, save_path=SAVE_PATH)
 
    print('Simulation finished')
    print(f"Collisions: {model.collisions}")
    print(f"Merges: {model.merges}")
    print(f"Bounces: {model.bounces}")
    print(f"Protoplanets left: {model.n_alive}")
 
 
if __name__ == "__main__":
    main()
