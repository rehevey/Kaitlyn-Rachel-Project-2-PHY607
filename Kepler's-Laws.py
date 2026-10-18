from vpython import *

#constants

G = 1
M = 1
m = 1
R = 1
v0 = 0.7 #if velocity is too low the sun will just pull the planet into itself (make it less than the escape velocity)

g1 = graph(xtitle="x", ytitle="y", width=400, height=400)

f1 = gcurve(color=color.blue, dot=True)     # planet curve
f2 = gcurve(color=color.red, dot = True)    # sun position
# f2.plot(0, 0)     # puts the sun at the oridin
f3 = gcurve(color=color.green)
f4 = gcurve(color=color.green)

g2 = graph(xtitle="t", ytitle="dA", width=400, height=400)
f5 = gcurve(color=color.blue)

rstart = vector(R,0,0)
p = m*vector(0,v0, 0)

t = 0
dt = 0.001 

r = vector(R,0,0) 

# ----- Checks 1st Law : Ellipse / v0 needs to be 1.2 or recalc a & b -----

# a= 1.785    #Pulled directly from the graph 
# b= 1.61 

# while t<15:
#     rate(1000)
#     F = -G*M*m*norm(r) / mag(r)**2  #r is vector positon, norm r is a unit vector built into vpython 
#     p = p + F*dt
#     r = r + p*dt/m 
#     f1.plot(r.x, r.y)
#     t = t + dt

# x = a
# dx = 0.01

# while x>-a:
#     yp = b * sqrt(1-x**2/a**2)
#     ym = -b * sqrt(1-x**2/a**2)
#     f3.plot(x - 0.79, yp)  # -0.79 is to get the ellipse to be centered at the origin of our planet orbit rather than the sun (0,0) 
#     f4.plot(x - 0.79, ym)
#     x = x - dx

# print('Done')
# while True:
#     rate(10) 


# ----- Checks Law 2 : Area of Sweep ----- 

# while t<15:
#     rate(1000)
#     F = -G*M*m*norm(r) / mag(r)**2  #r is vector positon, norm r is a unit vector built into vpython 
#     p = p + F*dt
#     r = r + p*dt/m 
#     dA = ((1)/(2*m) * mag(cross(r,p)))
#     f1.plot(r.x, r.y)
#     f5.plot(t, dA)
#     t = t + dt

# print('Done')
# while True:
#     rate(10) 

# ----- Check Law 3 :T^2 proprtional to b^3 ----- 

g3 = graph(xtitle="b^3", ytitle="T^2", width=300, height=400)
f6 = gcurve(color=color.purple)

def torbit(r0, v0):

    G = 1
    M = 1
    m = 1
    R = 1

    rstart = vector(r0,0,0)
    p = m*vector(0,v0, 0)

    t = 0
    dt = 0.001 

    r = vector(r0, 0, 0) # reassign r with r0 as first varriable instead of R so we can pass through different values easier

    orbit=True
    ymax = 0

    while orbit:

        F = -G*M*m*norm(r) / mag(r)**2  #r is vector positon, norm r is a unit vector built into vpython 
        p = p + F*dt
        r = r + p*dt/m 

        rdiff = r - rstart

        if t>1 and mag(rdiff)<0.1:
            orbit=False 

        if r.y>ymax:
            ymax = r.y

        t = t + dt

    return(t, ymax)

print(torbit(1,1)) 

v = 0.6
dv = 0.1
R = 1

while v<1.3:
    [T,b] = torbit(R, v)
    f6.plot(b**3, T**2)
    v = v + dv

print('Done')
while True:
    rate(10) 