#Needs to be a python class with at least two methods.
#Method suggestion: putting the protoplanets on the graph,
#assigning initial conditions to protoplanets 
#For at least one of the random samplings use Inverse CDF sampling. 
#For at least one of the random samplings use rejection sampling
#(suggest using this for the initial positions).

#We need a python class that makes a N number of points (protoplanets) in a circle surrounding a central star (sun, at the origin).
#Using the vpython package, make a list for all of the protoplanets ,then for each protoplanet in the list assign a mass, velocity, momentum, and force.

#Don't forget to include/account for the time and the time step.

#The sphere command can be used to make the visuals for the protoplanets and sun while also being able to set the initial position. 

import numpy as np
import matplotlib.pyplot as plt


class Particle_Source:
    def __init__(self,N, r_in=0.8, r_out=1.5, seed=42, m_min=-1e-8, m_max=1e-6,
                 alpha=2.5, sun_mass=1.0,G=1.0,r_soft=0.05, dt=0.01):
        self.N = N
        self.r_in = r_in #defines inner edge of orbit
        self.r_out = r_out #defines outer edge of orbit
        self.seed = seed #allows for reproducibility: if same numbers drawn --> same result!
        self.m_min = m_min  #defines lower range bound of protoplanet masses
        self.m_max = m_max  #defines upper range bound of protoplanet masses
        self.alpha = alpha #defines power-law exponent
        self.sun_mass = sun_mass #mass of sun!!
        self.G = G #gravitational constant
        self.r_soft = r_soft    #softening
        self.dt = dt  #time_step
        self.time = 0.0 #initial time, will increase with N individual steps.
        self.protoplanets = []

        self.M = np.zeros(N)
        self.pos = np.zeros((N,3))
        self.vel = np.zeros((N,3))
        self.F = np.zeros((N,3))
        self.p = np.zeros((N,3))

    def proto_list(self):
        #Goal = to create a list of protoplanets
        #Each protoplanet is assigned a random position, mass, velocity, momentum, and force
        

     def masses_inverse_CDF(self): #the masses of the protoplanets -->randomly sampled via inverse CDF sampling
        #with inverse CDF, one puts in a percentile & should spit out a desired value
        #X=F-1(U)
        #pdf = imf, k * m^-alpha
        U = np.random.uniform(0,1,self.N) #U = uniform distribution
        exp_factor = 1 - self.alpha #k


        # Do inverse CDF, m_max & m_min as bounds. 
        low_bound = self.m_min**exp_factor
        high_bound = self.m_max**exp_factor
        masses = ((low_bound + U) * (high_bound - low_bound)) ** (1 / exp_factor)

        return masses

    def initial_positions(self):
        #Sampling initial positions via rejection sampling:
        # generate (pseudo)independent samples from unnormalized pdf

        hits = []
        sigma_max = self.r_in**-1.5

        while len(hits) < self.N:
            x,y = self.rng.uniform(-self.r_out, self.r_out,2)
            r = np.sqrt(x**2 + y**2)
            U = self.rng.uniform(0, sigma_max)
            if self.r_in <= r <= self.r_out and U < r**-1.5:
                hits.append([x,y])

        return np.array(hits)


    def initial_conditions(self):

        self.mass = self.masses_inverse_CDF()
        xy = self.initial_positions()
        r = np.hypot(xy[:,0],xy[:,]) #the same function as r in initial positions func, maybe match?
        phi = np.arctan2(xy[:, 1], xy[:, 0])
        z = self.rng.normal(0.0, 0.01 * r)
 
        self.pos = np.column_stack([xy, z])
 
        v_circ = np.sqrt(self.G * self.star_mass / r)
        tangent = np.column_stack([-np.sin(phi), np.cos(phi),
                                   np.zeros(self.n)])
        dispersion = self.rng.normal(0.0, 0.01, (self.n, 3))
        self.vel = v_circ[:, None] * tangent + dispersion * v_circ[:, None]
 
        self.momentum = self.mass[:, None] * self.vel
        self.compute_forces()

 ##rest placed in troubleshoot file