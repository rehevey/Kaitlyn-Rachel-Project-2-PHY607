from vpython import *

g1 = graph(xtitle="t", ytitle="px", width=400, height=200)
fpA = gcurve(color=color.red)
fpB = gcurve(color=color.blue)
fpT = gcurve(color=color.purple)

g2 = graph(xtitle="t", ytitle="KE", width=400, height=200)
fKA = gcurve(color=color.red)
fKB = gcurve(color=color.blue)
fKT = gcurve(color=color.purple) 

m0 = 0.05   #mass
k = 50  #spring constant
R = 0.01    #radius

ballA = sphere(pos=vector(-4*R,0,0),radius=R, color=color.red, make_trail=True)
ballB = sphere(pos=vector(0,0,0),radius=R, color=color.blue, make_trail=True)

ballA.m = m0
ballB.m = m0
ballA.p = ballA.m * vector(0.1,0.05,0)
ballB.p = ballB.m * vector(0,0,0)

t = 0
dt = 0.01 

contact = False     # are they touching 

while t<20:
    rate(1000)
    FA = vector(0,0,0)
    FB = vector(0,0,0)

    r = ballB.pos - ballA.pos

    if mag(r)<2*R or contact:
        FA = -k*(2*R - mag(r))*norm(r)
        FB = -FA
        contact = True

    ballA.p = ballA.p + FA*dt
    ballB.p = ballB.p + FB*dt
    ballA.pos = ballA.pos + ballA.p*dt/ballA.m
    ballB.pos = ballB.pos + ballB.p*dt/ballB.m

    fpA.plot(t, ballA.p.x)
    fpB.plot(t, ballB.p.x)
    fpT.plot(t, ballA.p.x + ballB.p.x)

    KA = mag(ballA.p)**2/(2*ballA.m)
    KB = mag(ballB.p)**2/(2*ballB.m)
    KT = KA + KB
    fKA.plot(t, KA)
    fKB.plot(t, KB)
    fKT.plot(t, KT)

    t = t + dt 

print('Simulation finished')
while True:
    rate(10) 