import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe

# s (circumference) = 4a * integration from 0 to phi of sqrt(1 -k^2 * sin^2theta dtheta)
#^^^^^^This is called the complete elliptic integral of the second kind & is a non-elementary integral.

# if an ellipse has the width 2a & heigh 2b, area = pi * a * b!

A = 5
B = 2
N = 1000

k = math.sqrt(1 - (B**2/A**2)) #the eccentricity of the ellipse

exact_circumference = 4 * A * ellipe(k**2)
exact_area = math.pi * A * B

#upper integral limit. this will calculate 1/4 of the ellipse:
phi = math.pi/2 

def f(x):
    result = np.sqrt(1 - (k**2) * (np.sin(x)**2))
    return result

rand_x = np.random.uniform(0, phi, N) #generating the random x values
rand_y = np.random.uniform(0, 1, N) #generating the random y values

#Under_integrand = 
under_integrand = np.sum(rand_y < f(rand_x))

#estimate of the area under the curve
area = (phi * 1) * (under_integrand/N)

circumference = 4 * A * area




print (f"Monte Carlo Estimate of the Circumference of the Ellipse: {circumference}")
print (f"Circumference of the Ellipse (ANALYTICAL): {exact_circumference}")

print (f"Monte Carlo Estimate of the Area of the Ellipse: {area}")
print (f"Area of the Ellipse (ANALYTICAL): {exact_area}")

#------------------------------Uncertainty scales as N^(-1/2)------------------------------

#Statistical error of a monte carlo decreases as: 1/sqrt(N)
# N = number of independent samples

def area_estimates(N, reps):
    x_vals = np.random.uniform(0, A, (reps, N)) #generating the random x values
    y_vals = np.random.uniform(0, B, (reps, N)) #generating the random y values
    hits = (x_vals/A)**2 + (y_vals/B)**2 <=1 
    fraction = hits.mean(axis=1)
    estimates = 4 * A * B * fraction

    return estimates


N_samples = ([10, 30, 100, 300, 1000, 3000, 10000])
reps = 2000

sd_area = np.array([area_estimates(N,reps).std() for N in N_samples])
slope = np.polyfit(np.log(N_samples), np.log(sd_area), 1)[0]
print (f"Fitted slope = {slope:.3f} (expect -0.5)")


#Compare with the exact prediction
p = math.pi / 4
sigma = 4 * A * B * np.sqrt(p * (1 - p)) 

print(f"Sigma * sqrt(N): measured -- {sd_area * np.sqrt(N_samples)}")
print(f"Sigma * sqrt(N): exact -- {sigma}")

plt.loglog(N_samples, sd_area, 'o-', label='Measured STD')
plt.loglog(N_samples, sigma/np.sqrt(N_samples), 'k--', lw=0.8, label=r'$\sigma/\sqrt{N}$ (exact)')
plt.xlabel("N")
plt.ylabel("STD of area estimate")
plt.legend()
plt.show()

#Statistical uncertainty Vs. Number of Samples:

plt.loglog(N_samples,sd_area, 'o-', label='Measured STD')
plt.xlabel("N")
plt.ylabel("STD of area estimate")
plt.title("Statistical Uncertainty Vs. Number of Samples")
plt.legend()
plt.show()




    