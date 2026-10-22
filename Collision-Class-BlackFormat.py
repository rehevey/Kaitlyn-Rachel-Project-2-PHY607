"""
N-body planetesimal collision model with energy and momentum diagnostics.

A ring of small bodies orbits a fixed Sun. Bodies feel the Sun's gravity and
each other's (softened) gravity. The ODEs are integrated with SciPy's DOP853
solver, which is stopped by a terminal event whenever two bodies touch. Each
contact is then resolved as either a merger or an inelastic bounce, with the
merger probability set by the ratio of the relative speed to the escape speed.

The module also logs conserved-quantity diagnostics (mass, momentum, angular
momentum, kinetic, potential and total energy) as a validation check, tracks
the largest body, and provides matplotlib plots in place of VPython.

Units are AU, years and solar masses.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Gravitational constant in AU^3 / (Msun yr^2), and the (fixed) central mass
G = 4.0 * np.pi**2
Msun = 1.0


# =====================================================================
#  Physics model
# =====================================================================


class CollisionModel:
    """
    Gravitating protoplanets that merge or bounce on contact.

    State is held in numpy arrays with one row per body. The ODE state passed
    to ``solve_ivp`` is ``y = [x1, y1, z1, x2, ..., vx1, vy1, vz1, vx2, ...]``
    and ``dy/dt = [v, a(r)]``. Bodies that have merged away are dropped from
    ``y`` each time the integration restarts.

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Initial positions in AU.
    vel : array_like, shape (N, 3)
        Initial velocities in AU/yr.
    m : array_like, shape (N,)
        Masses in solar masses.
    radius : array_like, shape (N,)
        Collision radii in AU.
    m0 : float
        Reference mass for the radius-growth law (solar masses).
    rad0 : float
        Radius of a body of mass ``m0`` (AU).
    seed : int, optional
        Seed for the random number generator that decides merge/bounce.
    e_rest : float, optional
        Coefficient of restitution for bounces (default 0.5).
    eps : float, optional
        Gravitational softening length in AU. Defaults to ``0.5 * rad0``.
    rtol, atol : float, optional
        Relative and absolute tolerances passed to ``solve_ivp``.
    contact_tol : float, optional
        Relative tolerance used when testing whether two bodies overlap.
    separation : float, optional
        Relative gap enforced between a pair after a bounce.

    Attributes
    ----------
    pos, vel, m, radius : numpy.ndarray
        Current state of every body (including bodies that have merged away).
    alive : numpy.ndarray of bool, shape (N,)
        ``False`` for bodies that have merged into another body.
    n_bodies : int
        Number of bodies the model was created with.
    t : float
        Current simulation time in years.
    collisions, merges, bounces : int
        Running counts of collisions and of each outcome.
    collision_log : list of tuple
        One ``(t, v_rel / v_esc, merged)`` entry per collision.
    collision_effects : list of dict
        Change in each conserved quantity caused by every collision.
    cum_dE : float
        Cumulative energy change caused by collisions.
    cum_dL : numpy.ndarray, shape (3,)
        Cumulative angular-momentum change caused by collisions.
    history : list of dict
        Diagnostics recorded at the end of every ``advance`` call.
    """

    def __init__(
        self,
        pos,
        vel,
        m,
        radius,
        m0,
        rad0,
        seed=None,
        e_rest=0.5,
        eps=None,
        rtol=1e-9,
        atol=1e-12,
        contact_tol=1e-6,
        separation=1e-4,
    ):
        """Initialise the model; see the class docstring for the parameters."""
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.m = np.array(m, dtype=float)
        self.radius = np.array(radius, dtype=float)
        self.alive = np.ones(len(self.m), dtype=bool)  # every body starts "on"
        self.n_bodies = len(self.m)

        self.m0 = m0  # reference mass for the radius-growth law
        self.rad0 = rad0  # radius of a mass-m0 body
        self.e_rest = e_rest  # coefficient of restitution for bounces
        self.eps = 0.5 * rad0 if eps is None else eps  # softening length (AU)
        self.rtol = rtol
        self.atol = atol
        self.contact_tol = contact_tol  # relative tolerance for overlap test
        self.separation = separation  # relative gap enforced after a bounce

        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        # Counters for the number of collisions and of each outcome
        self.collisions = 0
        self.merges = 0
        self.bounces = 0

        # Logs used for the histogram and for validation
        self.collision_log = []
        self.collision_effects = []
        self.cum_dE = 0.0
        self.cum_dL = np.zeros(3)
        self.history = []
        self.record()  # the t = 0 entry

    # ---------- ODE system ----------
    @staticmethod
    def unpack(y, n):
        """
        Split the flat ODE state vector into positions and velocities.

        Parameters
        ----------
        y : numpy.ndarray, shape (6 * n,)
            Flat state ``[positions..., velocities...]``.
        n : int
            Number of bodies in the state.

        Returns
        -------
        pos : numpy.ndarray, shape (n, 3)
            Positions.
        vel : numpy.ndarray, shape (n, 3)
            Velocities.
        """
        # reshape un-smushes the flat 1D array that solve_ivp works with
        return y[: 3 * n].reshape(n, 3), y[3 * n :].reshape(n, 3)

    def rhs(self, t, y, mass):
        """
        Right-hand side of the equations of motion.

        Each body feels the (unsoftened) pull of the fixed Sun plus the
        Plummer-softened pull of every other body.

        Parameters
        ----------
        t : float
            Time (unused; the system is autonomous but ``solve_ivp`` needs it).
        y : numpy.ndarray, shape (6 * n,)
            Flat state vector.
        mass : numpy.ndarray, shape (n,)
            Masses of the bodies in ``y``.

        Returns
        -------
        numpy.ndarray, shape (6 * n,)
            Time derivative ``[velocities..., accelerations...]``.
        """
        n = len(mass)
        pos, vel = self.unpack(y, n)

        # Mutual gravity. d[i, j] = pos[j] - pos[i], so d[i, i] = 0 and a body
        # feels no force from itself. The shapes broadcast as:
        #   pos[None, :, :] -> (1, n, 3), indexed [0, j, k] = pos[j, k]
        #   pos[:, None, :] -> (n, 1, 3), indexed [i, 0, k] = pos[i, k]
        d = pos[None, :, :] - pos[:, None, :]
        s2 = np.einsum("ijk,ijk->ij", d, d) + self.eps**2
        acc = (G * mass[None, :, None] * d / s2[:, :, None] ** 1.5).sum(axis=1)

        # Pull of the Sun, which sits fixed at the origin
        r3 = np.linalg.norm(pos, axis=1) ** 3
        acc -= G * Msun * pos / r3[:, None]

        # ravel flattens back to 1D, which is the shape the solver expects
        return np.concatenate([vel.ravel(), acc.ravel()])

    def contact_event(self, t, y, mass, rad):
        """
        Smallest surface-to-surface gap over all pairs.

        The value crosses zero at the first contact, which is what the
        solver's event detection looks for.

        Parameters
        ----------
        t : float
            Time (unused).
        y : numpy.ndarray, shape (6 * n,)
            Flat state vector.
        mass : numpy.ndarray, shape (n,)
            Masses of the bodies in ``y`` (only used for their count).
        rad : numpy.ndarray, shape (n,)
            Collision radii of the bodies in ``y``.

        Returns
        -------
        float
            Minimum gap between any two surfaces (AU); 1.0 if fewer than two
            bodies remain.
        """
        n = len(mass)
        if n < 2:
            return 1.0
        pos, _ = self.unpack(y, n)
        dist = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        gap = dist - (rad[:, None] + rad[None, :])
        np.fill_diagonal(gap, np.inf)  # ignore each body's gap to itself
        return gap.min()

    def _make_event(self, rad):
        """
        Build the terminal contact event for the current set of radii.

        Parameters
        ----------
        rad : numpy.ndarray, shape (n,)
            Collision radii of the bodies being integrated.

        Returns
        -------
        callable
            Event function with ``terminal`` and ``direction`` attributes set.
        """

        # solve_ivp passes the same args to events as to rhs, so the radii are
        # captured by closure instead
        def ev(t, y, mass):
            """Return the minimum surface gap (see ``contact_event``)."""
            return self.contact_event(t, y, mass, rad)

        ev.terminal = True  # stop integrating at contact
        ev.direction = -1  # only trigger when the gap goes positive -> negative
        return ev

    # ---------- collision helpers ----------
    def vrel_over_vesc(self, a, b):
        """
        Relative speed of two bodies divided by their mutual escape speed.

        Parameters
        ----------
        a, b : int
            Indices of the two bodies.

        Returns
        -------
        float
            ``v_rel / v_esc`` with ``v_esc = sqrt(2 G (m_a + m_b) / (r_a + r_b))``.
        """
        v_rel = np.linalg.norm(self.vel[b] - self.vel[a])
        v_esc = np.sqrt(
            2 * G * (self.m[a] + self.m[b]) / (self.radius[a] + self.radius[b])
        )
        return v_rel / v_esc

    def merge_probability(self, a, b):
        """
        Probability that a collision between two bodies ends in a merger.

        Parameters
        ----------
        a, b : int
            Indices of the two bodies.

        Returns
        -------
        float
            ``1 / (1 + (v_rel / v_esc)**2)``.
        """
        return 1.0 / (1.0 + self.vrel_over_vesc(a, b) ** 2)

    def do_merge(self, a, b):
        """
        Merge body ``b`` into body ``a``.

        Mass and linear momentum are conserved. The new body sits at the
        centre of mass, its radius follows a constant-density growth law, and
        body ``b`` is marked dead.

        Parameters
        ----------
        a, b : int
            Indices of the surviving and the absorbed body.
        """
        m, pos, vel = self.m, self.pos, self.vel
        m_new = m[a] + m[b]
        pos[a] = (m[a] * pos[a] + m[b] * pos[b]) / m_new
        vel[a] = (m[a] * vel[a] + m[b] * vel[b]) / m_new  # momentum conserved
        m[a] = m_new
        self.radius[a] = self.rad0 * (m_new / self.m0) ** (1 / 3)
        self.alive[b] = False

    def do_bounce(self, a, b):
        """
        Bounce two bodies off each other with restitution ``e_rest``.

        The impulse acts along the line of centres and only if the bodies are
        still approaching. The pair is then pushed apart about their centre of
        mass so they no longer overlap.

        Parameters
        ----------
        a, b : int
            Indices of the two colliding bodies.
        """
        m, pos, vel, radius = self.m, self.pos, self.vel, self.radius
        n = pos[b] - pos[a]
        dist = np.linalg.norm(n)
        n = n / dist  # unit vector along the line of centres
        vn = np.dot(vel[b] - vel[a], n)
        if vn < 0:  # only if still approaching
            J = (1 + self.e_rest) * vn * m[a] * m[b] / (m[a] + m[b])
            vel[a] += J * n / m[a]
            vel[b] -= J * n / m[b]

        # Push the pair apart so they are no longer overlapping. Otherwise the
        # contact event would start the next integration at a negative gap.
        target = (radius[a] + radius[b]) * (1 + self.separation)
        shift = target - dist
        if shift > 0:
            pos[a] -= n * shift * m[b] / (m[a] + m[b])
            pos[b] += n * shift * m[a] / (m[a] + m[b])

    def find_overlap(self):
        """
        Find one overlapping pair of alive bodies.

        Returns
        -------
        tuple of int or None
            Indices ``(a, b)`` of an overlapping pair, or ``None`` if no pair
            overlaps.
        """
        idx = np.flatnonzero(self.alive)
        if len(idx) < 2:
            return None
        pos = self.pos[idx]
        dist = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        rr = self.radius[idx]
        contact = dist < (rr[:, None] + rr[None, :]) * (1 + self.contact_tol)
        ii, jj = np.nonzero(np.triu(contact, 1))  # upper triangle: each pair once
        if len(ii) == 0:
            return None
        return int(idx[ii[0]]), int(idx[jj[0]])

    def handle_collisions(self):
        """
        Resolve every overlapping pair as a merge or a bounce.

        For each collision the conserved quantities are measured before and
        after, and the ratio ``v_rel / v_esc`` is logged. The ratio must be
        measured before the velocities are changed by the collision.
        """
        for _ in range(4 * self.n_bodies):  # safety cap
            pair = self.find_overlap()
            if pair is None:
                return
            a, b = pair
            self.collisions += 1
            before = self.diagnostics()
            ratio = self.vrel_over_vesc(a, b)
            merged = self.rng.random() < 1.0 / (1.0 + ratio**2)
            self.collision_log.append((self.t, ratio, merged))
            if merged:
                self.do_merge(a, b)
                self.merges += 1
            else:
                self.do_bounce(a, b)
                self.bounces += 1
            self._log_collision_effect(before, self.diagnostics(), merged, ratio)

    # ---------- advance with solve_ivp ----------
    def advance(self, T):
        """
        Integrate all alive bodies forward, stopping at each contact.

        The solver is restarted after every collision because the number of
        bodies (and their radii) may have changed. Diagnostics are recorded
        once at the end.

        Parameters
        ----------
        T : float
            Time to advance in years.

        Raises
        ------
        RuntimeError
            If ``solve_ivp`` reports failure.
        """
        t_left = T  # time still to be integrated in this call
        for _ in range(10 * self.n_bodies):  # safety cap on restarts
            if t_left <= 1e-12:
                break
            idx = np.flatnonzero(self.alive)  # re-pack: n changes after merges
            n = len(idx)
            y0 = np.concatenate([self.pos[idx].ravel(), self.vel[idx].ravel()])
            ev = self._make_event(self.radius[idx].copy()) if n > 1 else None

            sol = solve_ivp(
                self.rhs,
                (0.0, t_left),
                y0,
                args=(self.m[idx],),
                method="DOP853",
                rtol=self.rtol,
                atol=self.atol,
                events=ev,
            )
            if not sol.success:
                raise RuntimeError(sol.message)

            self.pos[idx], self.vel[idx] = self.unpack(sol.y[:, -1], n)
            dt_done = sol.t[-1]
            t_left -= dt_done
            self.t += dt_done

            if sol.status == 1:  # the solver stopped at a contact
                self.handle_collisions()

        self.record()

    # ---------- diagnostics (validation) ----------
    def diagnostics(self):
        """
        Totals of the conserved quantities over the alive bodies.

        The Sun sits fixed at the origin and is not included in the momentum,
        angular momentum or kinetic energy. The potential energy includes the
        Sun and uses the same softening as :meth:`rhs`.

        Returns
        -------
        dict
            ``mass`` : total mass; ``m_max`` : largest mass; ``n_alive`` :
            number of alive bodies; ``P`` : total momentum (3-vector);
            ``L`` : total angular momentum about the Sun (3-vector);
            ``KE``, ``PE``, ``E`` : kinetic, potential and total energy;
            ``pscale`` : sum of the individual ``|p_i|``, used to normalise
            ``|P|``.
        """
        a = np.flatnonzero(self.alive)
        mass, pos, vel = self.m[a], self.pos[a], self.vel[a]
        momentum = (mass[:, None] * vel).sum(axis=0)
        ang_mom = (mass[:, None] * np.cross(pos, vel)).sum(axis=0)
        ke = 0.5 * (mass * np.einsum("ij,ij->i", vel, vel)).sum()

        # Potential energy: softened pairs (each counted once) plus the Sun
        d = pos[None, :, :] - pos[:, None, :]
        soft = np.sqrt(np.einsum("ijk,ijk->ij", d, d) + self.eps**2)
        pe_pair = -G * np.triu(mass[:, None] * mass[None, :] / soft, 1).sum()
        pe_sun = -G * Msun * (mass / np.linalg.norm(pos, axis=1)).sum()
        pe = pe_pair + pe_sun

        return dict(
            mass=mass.sum(),
            m_max=mass.max(),
            n_alive=len(a),
            P=momentum,
            L=ang_mom,
            KE=ke,
            PE=pe,
            E=ke + pe,
            pscale=(mass * np.linalg.norm(vel, axis=1)).sum(),
        )

    def _log_collision_effect(self, before, after, merged, ratio):
        """
        Store how one collision changed each conserved quantity.

        Parameters
        ----------
        before, after : dict
            Output of :meth:`diagnostics` just before and just after the
            collision was resolved.
        merged : bool
            Whether the collision was a merger (``True``) or a bounce.
        ratio : float
            ``v_rel / v_esc`` for the collision.
        """
        d_e = after["E"] - before["E"]
        d_l = after["L"] - before["L"]
        self.cum_dE += d_e
        self.cum_dL += d_l
        self.collision_effects.append(
            dict(
                t=self.t,
                merged=merged,
                ratio=ratio,
                dm=after["mass"] - before["mass"],
                dP=after["P"] - before["P"],
                dL=d_l,
                dKE=after["KE"] - before["KE"],
                dPE=after["PE"] - before["PE"],
                dE=d_e,
                pscale=before["pscale"],
            )
        )

    def record(self):
        """Append the current diagnostics, time and cumulative changes to history."""
        d = self.diagnostics()
        d["t"] = self.t
        d["dE_coll"] = self.cum_dE
        d["dL_coll"] = self.cum_dL.copy()
        self.history.append(d)

    # ---------- summary data ----------
    @property
    def n_alive(self):
        """int : Number of bodies that have not merged away."""
        return int(self.alive.sum())

    @property
    def largest_mass(self):
        """float : Mass of the most massive alive body (solar masses)."""
        return float(self.m[self.alive].max())

    def summary(self):
        """
        One-line status string.

        Returns
        -------
        str
            Time, collision counts, bodies remaining and largest mass.
        """
        return (
            f"t = {self.t:.1f} | collisions: {self.collisions} | "
            f"merges: {self.merges} | bounces: {self.bounces} | "
            f"protoplanets left: {self.n_alive} | "
            f"largest: {self.largest_mass / self.m0:.0f} m0"
        )


# =====================================================================
#  Initial conditions: swap this function for Kaitlyn's class.
#  All the model needs is pos (N,3), vel (N,3), m (N,), radius (N,).
# =====================================================================


def make_initial_conditions(
    n_bodies=100, m0=1e-7, rad0=0.015, r_in=0.8, r_out=1.5, seed=42
):
    """
    Create a thin ring of bodies on near-circular, near-coplanar orbits.

    The orbital plane is x-z, with y the (small) vertical direction.

    Parameters
    ----------
    n_bodies : int, optional
        Number of bodies.
    m0 : float, optional
        Mass of every body (solar masses).
    rad0 : float, optional
        Collision radius of every body (AU).
    r_in, r_out : float, optional
        Inner and outer edge of the ring (AU).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    pos : numpy.ndarray, shape (n_bodies, 3)
        Positions in AU.
    vel : numpy.ndarray, shape (n_bodies, 3)
        Velocities in AU/yr.
    m : numpy.ndarray, shape (n_bodies,)
        Masses in solar masses.
    radius : numpy.ndarray, shape (n_bodies,)
        Collision radii in AU.
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_in, r_out, n_bodies)
    th = rng.uniform(0, 2 * np.pi, n_bodies)
    pos = np.column_stack(
        (r * np.cos(th), 0.002 * r * rng.normal(size=n_bodies), r * np.sin(th))
    )

    # Circular speed with 0.5% scatter, directed along the tangent
    v_circ = np.sqrt(G * Msun / r)
    speed = v_circ * (1 + 0.005 * rng.normal(size=n_bodies))
    vel = (
        np.column_stack((np.sin(th), np.zeros(n_bodies), -np.cos(th)))
        * speed[:, None]
    )
    vel[:, 1] += 0.003 * v_circ * rng.normal(size=n_bodies)  # vertical scatter

    m = np.full(n_bodies, m0)
    radius = np.full(n_bodies, rad0)
    return pos, vel, m, radius


# =====================================================================
#  Running the model and plotting (display only)
# =====================================================================


def run_headless(model, t_end, frame_dt):
    """
    Advance the model to ``t_end`` with no graphics, printing progress.

    Parameters
    ----------
    model : CollisionModel
        Model to advance (modified in place).
    t_end : float
        Final time in years.
    frame_dt : float
        Time advanced per call to ``model.advance`` (years).
    """
    next_report = 0.0
    while model.t < t_end:
        model.advance(frame_dt)
        if model.t >= next_report:
            print(model.summary())
            next_report += 5.0  # report every 5 years


def histogram_counts(model, bins):
    """
    Bin the logged ``v_rel / v_esc`` values, split by collision outcome.

    Values outside the bin range are placed in the first or last bin.

    Parameters
    ----------
    model : CollisionModel
        Model whose ``collision_log`` is binned.
    bins : numpy.ndarray, shape (nbins + 1,)
        Bin edges.

    Returns
    -------
    merge_counts : numpy.ndarray, shape (nbins,)
        Number of merging collisions in each bin.
    bounce_counts : numpy.ndarray, shape (nbins,)
        Number of bouncing collisions in each bin.
    """
    if not model.collision_log:
        zeros = np.zeros(len(bins) - 1)
        return zeros, zeros.copy()
    ratio = np.array([c[1] for c in model.collision_log])
    merged = np.array([c[2] for c in model.collision_log], dtype=bool)
    # Clip so out-of-range values land in the end bins
    ratio = np.clip(ratio, bins[0], bins[-1] * (1 - 1e-12))
    return (
        np.histogram(ratio[merged], bins=bins)[0],
        np.histogram(ratio[~merged], bins=bins)[0],
    )


def style_hist_axis(ax, xmin, xmax):
    """
    Apply the dark theme, log x-axis and labels to a histogram axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style (modified in place).
    xmin, xmax : float
        Limits of the (logarithmic) x-axis.
    """
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"$v_{rel}/v_{esc}$ at collision", color="white")
    ax.set_ylabel("collisions", color="white")
    ax.set_title("Collision speeds", color="white")


def plot_histogram(model, xmin=0.1, xmax=100.0, nbins=40):
    """
    Show a static histogram of ``v_rel / v_esc`` for the finished run.

    Merges are drawn in green with bounces stacked on top in red.

    Parameters
    ----------
    model : CollisionModel
        Model whose collisions are plotted.
    xmin, xmax : float, optional
        Range of the logarithmic x-axis.
    nbins : int, optional
        Number of logarithmically spaced bins.
    """
    import matplotlib.pyplot as plt

    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    merge_counts, bounce_counts = histogram_counts(model, bins)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("black")
    widths = np.diff(bins)
    ax.bar(
        bins[:-1],
        merge_counts,
        width=widths,
        align="edge",
        color="limegreen",
        label="merge",
    )
    ax.bar(
        bins[:-1],
        bounce_counts,
        width=widths,
        align="edge",
        bottom=merge_counts,
        color="tomato",
        label="bounce",
    )
    style_hist_axis(ax, xmin, xmax)
    ax.legend(facecolor="black", labelcolor="white")
    plt.show()


def history_arrays(model):
    """
    Convert the list of diagnostics dictionaries into arrays.

    Parameters
    ----------
    model : CollisionModel
        Model whose ``history`` is converted.

    Returns
    -------
    dict of numpy.ndarray
        One array per diagnostic, with time along the first axis.
    """
    keys = model.history[0].keys()
    return {k: np.array([h[k] for h in model.history]) for k in keys}


def print_validation(model):
    """
    Print a text summary of how well the conserved quantities are conserved.

    "Corrected" values are the raw totals minus the cumulative change caused
    by collisions. They isolate integrator error from the (intentional)
    dissipation in the collision rules.

    Parameters
    ----------
    model : CollisionModel
        Model that has been advanced (needs a populated ``history``).
    """
    h = history_arrays(model)
    e0, m0_tot = h["E"][0], h["mass"][0]
    l0 = np.linalg.norm(h["L"][0])
    d_mass = np.abs(h["mass"] - m0_tot) / m0_tot
    d_e_raw = np.abs(h["E"] - e0) / abs(e0)
    d_e_cor = np.abs(h["E"] - h["dE_coll"] - e0) / abs(e0)
    d_l_raw = np.linalg.norm(h["L"] - h["L"][0], axis=1) / l0
    d_l_cor = np.linalg.norm(h["L"] - h["dL_coll"] - h["L"][0], axis=1) / l0

    print("\n===== Validation =====")
    print(
        f"Initial: M = {m0_tot:.6e} Msun | KE = {h['KE'][0]:.6e} | "
        f"PE = {h['PE'][0]:.6e} | E = {e0:.6e} | |L| = {l0:.6e}"
        "   (AU, yr, Msun)"
    )
    print(
        f"Final:   M = {h['mass'][-1]:.6e} | KE = {h['KE'][-1]:.6e} | "
        f"PE = {h['PE'][-1]:.6e} | E = {h['E'][-1]:.6e} | "
        f"|L| = {np.linalg.norm(h['L'][-1]):.6e}"
    )
    print(f"Max relative drift  mass:                {d_mass.max():.2e}")
    print(
        f"Max relative drift  energy   (raw):      {d_e_raw.max():.2e}"
        f"   (corrected for collisions): {d_e_cor.max():.2e}"
    )
    print(
        f"Max relative drift  ang. mom. (raw):     {d_l_raw.max():.2e}"
        f"   (corrected for collisions): {d_l_cor.max():.2e}"
    )

    effects = model.collision_effects
    if not effects:
        print("No collisions occurred, so raw and corrected values are identical.")
        return
    max_dm = max(abs(e["dm"]) for e in effects) / m0_tot
    max_dp = max(np.linalg.norm(e["dP"]) / e["pscale"] for e in effects)
    print(
        f"Across {len(effects)} collisions:  max |dM|/M0 = {max_dm:.2e},  "
        f"max |dP|/sum|p| = {max_dp:.2e}   (both should be ~machine precision)"
    )
    for label, flag in (("merges", True), ("bounces", False)):
        sel = [e for e in effects if e["merged"] == flag]
        if sel:
            print(
                f"  {label:8s} n = {len(sel):4d} | "
                f"sum dKE = {sum(e['dKE'] for e in sel):+.3e} | "
                f"sum dPE = {sum(e['dPE'] for e in sel):+.3e} | "
                f"sum dE = {sum(e['dE'] for e in sel):+.3e} | "
                f"sum |dL| = {sum(np.linalg.norm(e['dL']) for e in sel):.3e}"
            )
    print(
        f"Total collisional energy change: {model.cum_dE:+.3e}  "
        f"({model.cum_dE / abs(e0):+.2e} of |E0|)"
    )
    total_dl = np.linalg.norm(model.cum_dL)
    print(
        f"Total collisional |dL|:          {total_dl:.3e}  "
        f"({total_dl / l0:.2e} of |L0|)"
    )


def plot_validation(model, save_path=None):
    """
    Plot mass, momentum, energies and angular momentum against time.

    Parameters
    ----------
    model : CollisionModel
        Model that has been advanced (needs a populated ``history``).
    save_path : str, optional
        If given, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt

    h = history_arrays(model)
    t = h["t"]
    floor = 1e-17  # keeps exact zeros visible on a log axis
    e0, m0_tot = h["E"][0], h["mass"][0]
    l0 = np.linalg.norm(h["L"][0])
    e_cor = h["E"] - h["dE_coll"]  # energy with collisional change removed
    l_cor = h["L"] - h["dL_coll"]  # angular momentum likewise

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    ax = axs[0, 0]
    ax.plot(t, (h["mass"] - m0_tot) / m0_tot)
    ax.set_title("Total mass: (M - M0)/M0")
    if np.abs(h["mass"] - m0_tot).max() / m0_tot < 1e-10:
        ax.set_ylim(-1e-10, 1e-10)  # avoid autoscaling pure round-off noise

    ax = axs[0, 1]
    ax.plot(t, np.linalg.norm(h["P"], axis=1) / h["pscale"])
    ax.set_title(
        r"Momentum: $|P| / \Sigma|p_i|$"
        "\n(not conserved: fixed Sun is an external force)",
        fontsize=10,
    )

    ax = axs[0, 2]
    ax.plot(t, h["KE"] / h["KE"][0], label="KE / KE0")
    ax.plot(t, h["PE"] / h["PE"][0], label="PE / PE0")
    ax.set_title("Kinetic and potential energy")
    ax.legend()

    ax = axs[1, 0]
    ax.semilogy(t, np.abs(h["E"] - e0) / abs(e0) + floor, label="raw")
    ax.semilogy(
        t,
        np.abs(e_cor - e0) / abs(e0) + floor,
        label="corrected for collisions",
    )
    ax.set_title("Total energy: |E - E0| / |E0|")
    ax.legend()

    ax = axs[1, 1]
    ax.semilogy(
        t,
        np.linalg.norm(h["L"] - h["L"][0], axis=1) / l0 + floor,
        label="raw",
    )
    ax.semilogy(
        t,
        np.linalg.norm(l_cor - h["L"][0], axis=1) / l0 + floor,
        label="corrected for collisions",
    )
    ax.set_title("Angular momentum: |L - L0| / |L0|")
    ax.legend()

    ax = axs[1, 2]
    ax.step(
        t,
        h["dE_coll"] / abs(e0),
        where="post",
        label=r"cumulative $\Delta E$ / |E0|",
    )
    ax.step(
        t,
        np.linalg.norm(h["dL_coll"], axis=1) / l0,
        where="post",
        label=r"cumulative |$\Delta L$| / |L0|",
    )
    ax.set_title("Change caused by collisions")
    ax.legend()

    for ax in axs.ravel():
        ax.set_xlabel("t (yr)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    else:
        plt.show()


def plot_growth(model, save_path=None):
    """
    Plot the largest body's mass and the number of bodies against time.

    Parameters
    ----------
    model : CollisionModel
        Model that has been advanced (needs a populated ``history``).
    save_path : str, optional
        If given, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt

    h = history_arrays(model)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(h["t"], h["m_max"] / model.m0, where="post", color="tab:blue")
    ax.set_xlabel("t (yr)")
    ax.set_ylabel(r"largest mass / $m_0$", color="tab:blue")
    ax.set_title("Growth of the largest body")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.step(h["t"], h["n_alive"], where="post", color="tab:orange")
    ax2.set_ylabel("bodies remaining", color="tab:orange")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    else:
        plt.show()


def check_integrator(t_end=10, frame_dt=0.05, n_bodies=100, seed=42):
    """
    Run with collisions effectively off to test the integrator on its own.

    With tiny radii no contacts occur, so energy and angular momentum should
    be conserved by the integrator alone.

    Parameters
    ----------
    t_end : float, optional
        Final time in years.
    frame_dt : float, optional
        Time advanced per call to ``model.advance`` (years).
    n_bodies : int, optional
        Number of bodies.
    seed : int, optional
        Seed for the initial conditions.

    Returns
    -------
    CollisionModel
        The advanced model (its ``history`` can be plotted).
    """
    pos, vel, m, _ = make_initial_conditions(
        n_bodies=n_bodies, m0=1e-7, rad0=0.015, seed=seed
    )
    model = CollisionModel(
        pos,
        vel,
        m,
        np.full(n_bodies, 1e-9),
        m0=1e-7,
        rad0=1e-9,
        seed=seed,
        eps=0.0075,
    )
    run_headless(model, t_end, frame_dt)
    print_validation(model)
    return model


def run_animation(
    model,
    t_end,
    frame_dt,
    view_range=1.8,
    save_path=None,
    steps_per_frame=1,
    hist_xmin=0.1,
    hist_xmax=100.0,
    hist_bins=40,
):
    """
    Animate the orbits alongside a live ``v_rel / v_esc`` histogram.

    The left panel is a top-down (x-z) view with marker sizes scaled to each
    body's physical radius. The right panel shows merges (green) with bounces
    (red) stacked on top, updated every frame.

    Parameters
    ----------
    model : CollisionModel
        Model to advance (modified in place).
    t_end : float
        Final time in years.
    frame_dt : float
        Time advanced per ``model.advance`` call (years).
    view_range : float, optional
        Half-width of the orbit view in AU.
    save_path : str, optional
        If given, write the animation to this file (``.gif``, or ``.mp4`` with
        ffmpeg) instead of opening a window.
    steps_per_frame : int, optional
        Number of ``advance`` calls per displayed frame.
    hist_xmin, hist_xmax : float, optional
        Range of the logarithmic histogram axis.
    hist_bins : int, optional
        Number of logarithmically spaced histogram bins.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (keep a reference so it is not garbage collected).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.patch.set_facecolor("black")

    # Orbit panel
    ax.set_facecolor("black")
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("N-body (SciPy DOP853 + collision events)", color="white")
    ax.scatter([0], [0], s=150, color="yellow", zorder=3)  # the Sun
    bodies = ax.scatter([], [], color="deepskyblue", zorder=2)
    stats = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, color="white", va="top", fontsize=9
    )

    # Histogram panel: merges (green) with bounces (red) stacked on top
    bins = np.logspace(np.log10(hist_xmin), np.log10(hist_xmax), hist_bins + 1)
    widths = np.diff(bins)
    zeros = np.zeros(hist_bins)
    merge_bars = axh.bar(
        bins[:-1], zeros, width=widths, align="edge", color="limegreen", label="merge"
    )
    bounce_bars = axh.bar(
        bins[:-1],
        zeros,
        width=widths,
        align="edge",
        bottom=zeros,
        color="tomato",
        label="bounce",
    )
    style_hist_axis(axh, hist_xmin, hist_xmax)
    axh.legend(facecolor="black", labelcolor="white", loc="upper right")

    def marker_sizes(radii):
        """
        Convert physical radii to scatter areas so sizes track real size.

        Parameters
        ----------
        radii : numpy.ndarray
            Collision radii in AU.

        Returns
        -------
        numpy.ndarray
            Marker areas in points squared (at least 2 points across).
        """
        fig.canvas.draw_idle()
        width_pts = ax.get_window_extent().width * 72 / fig.dpi
        pts_per_au = width_pts / (2 * view_range)
        diam_pts = 2 * radii * pts_per_au
        return np.maximum(diam_pts, 2.0) ** 2

    def update_hist():
        """Refresh the heights of the stacked histogram bars."""
        merge_counts, bounce_counts = histogram_counts(model, bins)
        for bar, height in zip(merge_bars, merge_counts):
            bar.set_height(height)
        for bar, height, base in zip(bounce_bars, bounce_counts, merge_counts):
            bar.set_height(height)
            bar.set_y(base)  # stack on top of the merge bar
        axh.set_ylim(0, max(5, (merge_counts + bounce_counts).max() * 1.1))

    def update(frame):
        """
        Advance the model and redraw both panels.

        Parameters
        ----------
        frame : object
            Frame value supplied by ``FuncAnimation`` (unused).

        Returns
        -------
        tuple
            The artists that were updated.
        """
        for _ in range(steps_per_frame):
            model.advance(frame_dt)
        alive = np.flatnonzero(model.alive)
        bodies.set_offsets(model.pos[alive][:, [0, 2]])  # x vs z = orbital plane
        bodies.set_sizes(marker_sizes(model.radius[alive]))
        stats.set_text(model.summary())
        update_hist()
        return bodies, stats

    def frames():
        """Yield frames until the simulation reaches ``t_end``."""
        while model.t < t_end:
            yield model.t

    fig.tight_layout()
    if save_path:
        # A saved animation needs a fixed number of frames
        n_frames = int(np.ceil(t_end / (frame_dt * steps_per_frame)))
        anim = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False)
        anim.save(save_path, fps=30, dpi=100)
        print(f"Saved {save_path}")
    else:
        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=16,
            blit=False,
            cache_frame_data=False,
            repeat=False,
        )
        plt.show()
    return anim


def main():
    """Set parameters, run the simulation and show the requested plots."""
    # ---------- parameters ----------
    n_bodies = 100
    m0 = 1e-7  # planetesimal mass (solar masses)
    rad0 = 0.015  # enhanced collision radius (AU)
    t_end = 100  # simulated years
    frame_dt = 0.05  # years advanced per animation frame (solver subdivides)

    mode = "animate"  # "animate" | "headless" | "check_integrator"
    validate = True  # print the validation summary and show its figure
    validation_plot = None  # e.g. "validation.png" to save instead of showing
    plot_growth_curve = True  # show the largest-mass-vs-time figure afterwards
    growth_plot = None  # e.g. "growth.png" to save instead of showing
    save_path = None  # e.g. "collisions.gif" to write the animation to a file

    if mode == "check_integrator":
        model = check_integrator(t_end=min(t_end, 20), frame_dt=frame_dt)
        plot_validation(model, save_path=validation_plot)
        return

    # ---------- model ----------
    pos, vel, m, radius = make_initial_conditions(
        n_bodies=n_bodies, m0=m0, rad0=rad0, seed=42
    )
    model = CollisionModel(
        pos, vel, m, radius, m0=m0, rad0=rad0, seed=42, e_rest=0.5, eps=0.5 * rad0
    )

    # ---------- run ----------
    if mode == "headless":
        run_headless(model, t_end, frame_dt)
        plot_histogram(model)
    else:
        run_animation(model, t_end, frame_dt, save_path=save_path)

    print("Simulation finished")
    print(f"Collisions: {model.collisions}")
    print(f"Merges: {model.merges}")
    print(f"Bounces: {model.bounces}")
    print(f"Protoplanets left: {model.n_alive}")
    print(
        f"Largest body: {model.largest_mass / model.m0:.0f} m0 "
        f"({model.largest_mass:.3e} Msun)"
    )

    # ---------- results ----------
    if plot_growth_curve:
        plot_growth(model, save_path=growth_plot)

    if validate:
        print_validation(model)
        plot_validation(model, save_path=validation_plot)


if __name__ == "__main__":
    main()