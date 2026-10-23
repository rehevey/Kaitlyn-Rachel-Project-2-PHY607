import numpy as np

## Units: AU, years, solar masses

G = 4.0 * np.pi**2  # AU^3 / (Msun yr^2)
Msun = 1.0


class ParticleSource:

    def __init__(self, N=100, r_in=0.8, r_out=1.5, seed=42, m_min=1e-8,
                 m_max=1e-6, alpha=2.5, m0=1e-7, rad0=0.015):
        
        self.N = N

        self.r_in = r_in  # defines inner edge of orbit
        self.r_out = r_out  # defines outer edge of orbit

        self.m_min = m_min  # defines lower range bound of protoplanet masses
        self.m_max = m_max  # defines upper range bound of protoplanet masses
        self.alpha = alpha  # defines power-law exponent, needed for inverse CDF

        self.m0 = m0  # reference mass for the radius-growth law
        self.rad0 = rad0  # radius of a body of mass m0

        # allows for reproducibility: if same numbers drawn --> same result!
        self.rng = np.random.default_rng(seed)

        self.M = self.sample_masses()  # inverse CDF
        self.pos = self.sample_positions()  # rejection sampling
        self.vel = self.circular_velocities()
        self.radius = self.rad0 * (self.M / self.m0) ** (1 / 3)
        self.p = self.M[:, None] * self.vel

    def sample_masses(self):
        # the masses of the protoplanets --> randomly sampled via inverse CDF
        # sampling with inverse CDF: one puts in a percentile & should spit out a desired value
        U = self.rng.uniform(0, 1, self.N) 
        k = 1.0 - self.alpha
        lo, hi = self.m_min**k, self.m_max**k
        return (lo + U * (hi - lo)) ** (1.0 / k)

    def sample_positions(self):
        # Sampling initial positions via rejection sampling:
        # generate (pseudo)independent samples from unnormalized pdf
        # positions in AU
        pts, sigma_max = [], self.r_in ** -1.5
        while len(pts) < self.N:
            # samples from uniform distribution, along x & z
            x, z = self.rng.uniform(-self.r_out, self.r_out, 2)
            r = np.hypot(x, z)
            if self.r_in <= r <= self.r_out and self.rng.uniform(0, sigma_max) < r ** -1.5:
                pts.append([x, 0.002 * r * self.rng.normal(), z])
        return np.array(pts)

    def circular_velocities(self):
        # the sun sits at the origin, so each body moves tangentially
        # with the circular speed sqrt(GM/r)
        x, z = self.pos[:, 0], self.pos[:, 2]
        r = np.hypot(x, z)
        vc = np.sqrt(G * Msun / r)
        return np.column_stack([vc * z / r, np.zeros(self.N), -vc * x / r])

    def model_inputs(self):
        return self.pos, self.vel, self.M, self.radius


def single_body(a, ecc=0.0):
    v = np.sqrt(G * Msun / a) * np.sqrt(1 - ecc)
    # a = starting distance. the true semi-major axis is a / (1 + ecc)
    return (np.array([[a, 0, 0]]), np.array([[0, 0, -v]]),
            np.array([1e-12]), np.array([1e-9]))

