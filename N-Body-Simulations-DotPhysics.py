from vpython import *
import random

scene = canvas(title="N-body", width=900, height=700, background=color.black)

#Give in constants 

G = 1
M = 1 # total mass
R = 1
rsoft = 0.05
w = vector(0, 0.05, 0)

#Counter numbr n and a count cap N 

n = 0
N = 10 #Starting with only 5 masses 

stars = []  # make an empty list for stars (masses) 

while n<N: #We want to go though and make random vectors 
    rt = R*vector(2*random.random()-1, 2*random.random()-1, 2*random.random()-1)
    # random.random retuns a number between 0 and 1, multipy by two to get btween 0 and 2, subtract 1 to get between -1 and 1
    # makes a -R to R box for all of the stars to go into 
    stars = stars + [sphere(pos=rt, radius=R/30, make_trail=True, retain=100)]  #train and retain are for the visual representation 
    n = n + 1

# for star in stars:
#     print(star.pos)

# for i in range(len(stars)):
#     print (stars[i].pos)

for star in stars:
    star.m = M/N   #give each star a mass, in this case the same mass 
    star.p = star.m*cross(w, vector(star.pos.x, 0, star.pos.z))    #velocity including rotational motion, we don't car about the y because w is around the y axis
    # star.p = star.m * vector(0.1*(2*random()-1),0,0)    #giving the stars a velocity (in only one direction)
    star.F = vector(0,0,0)  #the net force on that star 

t = 0   # time value
dt = 0.01   # time step 

while t<100:
    rate(100)   # limits the rate to 100 calculations per second 
    for star in stars:
        star.F = vector(0,0,0)      #starts net force at zero
    for i in range(len(stars)):         
        for j in range(len(stars)):     #double loop goes though all of the combinations of i and j (12, 13, 23, 34, 45, etc.) 
            if i!=j:  # as long as i and j are difference continue (the force of 1 on 1 or 2 on 2 would be infinate so we don't want to put that in our calculations)
                rji = stars[i].pos - stars[j].pos   #position of star i and postion of star j
                stars[i].F = stars[i].F -  G*stars[i].m*stars[j].m*norm(rji)/(mag(rji)**2 + rsoft**2)   #forces on star i; current force on i minus the gravity times the mass of stars i and j and norm/mag(rji) accounts of the vector between the stars

    for star in stars:
        star.p = star.p + star.F*dt     #update momentum/velocity 
        star.pos = star.pos + star.p*dt/star.m  #update star position

    t = t + dt  #update time
    
print('Simulation finished')
while True:
    rate(10)
