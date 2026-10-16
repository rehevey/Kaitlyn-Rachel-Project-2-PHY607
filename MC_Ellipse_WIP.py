import random
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# use the function e^(-x) on 0 to 1

#integrate it in two ways 
    #mean value: average f(x) over uniform smaples, times the interval 
    #hit or miss: sample a box and count what lands under the curve 

#estimate the unvertainty for each way 
    #run a bunch of times to get different values then th edistribution of those errors is the unertainly (standard deviation = 1/(sqrt n))
    #verify it scales as N^(-1/2) 

#instead of integrating -- draw samples from it 
    #inverse CDF way 
    #rejection sampling from a uniform way 


#Step 1 -- Set-up 

#Define the integral 

a = 0
b = 1 #integration limits a = lower , b = upper

N = 1000 #the number of points 


def func(x):
    return np.exp(-x)

#Random Array 

xrand = np.zeros(N)

for i in range(len(xrand)): 
    xrand[i] = random.uniform(a,b) #will pull N number of points of random value from between a and b with uniform distibution 


#Mean Value Solution 

mval_est = (b-a)*np.mean(func(xrand))
print("Monte Carlo Estimate:",mval_est)

#Ploting MC Mean Value  

repeats = 1000
mv_estimates = []

for i in range(repeats):
    xsamples = np.random.uniform(a , b , N)
    mv_estimates.append((b-a)*np.mean(func(xsamples)))

axs[0, 0].hist(mv_estimates, bins=30, edgecolor='black')
axs[0, 0].set_title("Mean Value Estimates")
axs[0, 0].set_xlabel("Estimate")
axs[0, 0].set_ylabel("Frequency")



#Hit or Miss Solution 

hits = 0 
misses = 0 #initial values 
n = 1000

#a set of empy lists to keep track of coordinates 
xH = []
yH = []
xM = []
yM = []

#main loop 
for i in range(n):
    xcord = np.random.uniform() #gernate the x coordinate 
    ycord = np.random.uniform() #generate the y coordinate

    if ycord <= func(xcord): #if the y coordinate is below the function line 
        hits += 1 #add an entry to the hit list
        xH = xH + [xcord]
        yH = yH + [ycord] 
    else: #if the y coordinate is above the function line 
        misses += 1 #add an entry to the misses list
        xM = xM + [xcord]
        yM = yM + [ycord]

hm_est = hits/n #estimates the evalue of the integral 
print('The Hit & Miss estimate is', hm_est)

#Ploting MC Hit or Miss   

hm_reps = 1000
n = 1000
hm_estimates = []

for i in range(hm_reps):
    xcordG = np.random.uniform(0, 1, n)
    ycordG = np.random.uniform(0, 1, n)

    hit_mask = ycordG <= func(xcordG) # boolean array, element-wise comparison
    hits = hit_mask.sum() 

    hm_estimates.append(hits/n)

axs[0, 1].hist(hm_estimates, bins=30, edgecolor='black')
axs[0, 1].set_title("Hit and Miss Estimates")
axs[0, 1].set_xlabel("Estimate")
axs[0, 1].set_ylabel("Frequency")



# Uncertainty Calculations 

exact = 1 - np.exp(-1) # analytical answer 
# integration from 0 to 1 of e ^ -x which equals 1 - e^-x which is about 0.632

# the two estimators 

def mean_value(N, reps):
    x_UM = np.random.uniform(a, b, size= (reps, N))
    return (b - a) * func(x_UM).mean(axis=1) 

def hit_or_miss(n, reps):
    x_UHm = np.random.uniform(0, 1, size=(reps, n))
    y_UHm = np.random.uniform(0, 1, size=(reps, n))
    return (y_UHm <= func(x_UHm)).mean(axis=1)

# uncertainty and N^-1/2 scaling 

Ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
reps = 2000
sd_mv = np.array([mean_value(N, reps).std() for N in Ns])
sd_hm = np.array([hit_or_miss(n, reps).std() for n in Ns])

for name, sd in [("mean value", sd_mv), ("hit or miss", sd_hm)]:
    slope = np.polyfit(np.log(Ns), np.log(sd), 1)[0]
    print(f"{name}: fitted slope = {slope:.3f} (expect -0.5)")

# exact per-sample standard deviations for comparison

sig_mv = np.sqrt((1 - np.exp(-2)) / 2 - exact**2)   # ~0.181
sig_hm = np.sqrt(exact * (1 - exact))               # ~0.482

print("sigma*sqrt(N): measured -- Mean Value ", sd_mv[-1]*np.sqrt(Ns[-1]),"Hit or Miss ", sd_hm[-1]*np.sqrt(Ns[-1]))
print("sigma*sqrt(N): exact -- Mean Value",  sig_mv,"Hit or Miss ", sig_hm)
print("variance ratio (hm/mv):", (sig_hm/sig_mv)**2)

axs[0, 2].loglog(Ns, sd_mv, 'o-', label='mean value')
axs[0, 2].loglog(Ns, sd_hm, 's-', label='hit or miss')
axs[0, 2].loglog(Ns, sig_mv/np.sqrt(Ns), 'k--', lw=0.8, label=r'$\sigma/\sqrt{N}$ (exact)')
axs[0, 2].loglog(Ns, sig_hm/np.sqrt(Ns), 'k--', lw=0.8)
axs[0, 2].set_xlabel("N")
axs[0, 2].set_ylabel("std of estimate")
axs[0, 2].legend()



#Sampling from the PDF  p(x) = e^-x / (1 - e^-1) on [0,1]
M = 1000000

def pdf(x):
    return func(x) / exact

def cdf(x):
    return (1 - np.exp(-x)) / exact 

#Inverse CDF  F(x) = u  ->  x = -ln(1 - u(1 - e^-1))

u = np.random.uniform(size=M)
x_inv = -np.log(1 - u * exact)

#Rejection from a uniform proposal: max of f = 1

xp = np.random.uniform(size=2*M)
yp = np.random.uniform(size=2*M)
x_rej = xp[yp <= func(xp)][:M]

print("rejection acceptance rate: ", (yp <= func(xp)).mean(), "(expected", exact, ")")

#check samples follow the PDF

grid = np.linspace(0, 1, 200)

for a_, s, t in zip([axs[1, 0], axs[1, 1]], [x_inv, x_rej], ["inverse CDF", "rejection"]):
    a_.hist(s, bins=40, density=True, alpha=0.6, edgecolor='black')
    a_.plot(grid, pdf(grid), 'r', label='exact pdf')
    a_.set_title(t)
    a_.set_xlabel("x")
    a_.legend()

for name, s in [("inverse CDF", x_inv), ("rejection", x_rej)]:
    print(name, "KS p-value:", stats.kstest(s, cdf).pvalue)


axs[1, 2].axis('off')   # the 6th slot is empty

plt.tight_layout()
plt.show()