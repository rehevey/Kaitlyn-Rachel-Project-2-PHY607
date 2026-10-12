import random
import numpy as np
import matplotlib.pyplot as plt 

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

mval_est = (b-a)*np.mean((xrand))
print("Monte Carlo Estimate:",mval_est)

#Ploting MC Mean Value  

repeats = 1000
mv_estimates = []

for i in range(repeats):
    xsamples = np.random.uniform(a , b , N)
    mv_estimates.append((b-a)*np.mean(xsamples))

plt.hist(mv_estimates, bins=30, edgecolor='black')
plt.title("Monte Carlo Estimate Distribution (Mean Value)")
plt.xlabel("Estimate")
plt.ylabel("Frequency")
plt.show()


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

plt.hist(hm_estimates, bins=30, edgecolor='black')
plt.title("Monte Carlo Estimate Distribution (Hit and Miss)")
plt.xlabel("Estimate")
plt.ylabel("Frequency")
plt.show()