from vpython import *

g1 = graph(xtitle="t", ytitle="Lz", width=400, height=400)
f1 = gcurve(color=color.blue, label="ball 1")
f2 =gcurve(color=color.red, label="pA")
fT = gcurve(color=color.purple,label="total")

m = 1
R = 1
F0 = 1
v = sqrt(F0*R/m)

pA = sphere(pos=vector(0, 0, 0), radius=R/20, make_trail=True)
pA.m = 20
pB = sphere(pos=vector(0, -1.5*R,0), radius=R/20)
ball1 = sphere(pos=vector(R,0 ,0), radius=R/15, color = color.yellow, make_trail=True)

ball1.m = m
ball1.p = ball1.m*vector(0, v, 0)

pA.p= -ball1.p  #total momentum is 0

t = 0
dt = 0.01

while t<7:
    rate(100)
    r1A = ball1.pos - pA.pos
    r1B = ball1.pos - pB.pos
    F = -F0*norm(r1A)
    ball1.p = ball1.p + F*dt 
    pA.p = pA.p - F*dt
    ball1.pos = ball1.pos + ball1.p*dt/ball1.m
    pA.pos = pA.pos + pA.p*dt/pA.m 
    rAB = pA.pos - pB.pos   #vector from pB to pA 
    L1 = cross(r1B,ball1.p)   # change 1st position between ball1.pos for pA and r1B for pB
    LA = cross(rAB, pA.p)    # 1st position pA.pos for momentum consevation btwen pA & ball, rAB for pB & ball
    Ltot = L1 + LA
    f1.plot(t,L1.z)
    f2.plot(t,LA.z)
    fT.plot(t, Ltot.z)
    t = t + dt

print('Simulation finished')
while True:
    rate(10)

# When looking at an opject orbiting the sun, the sun is also moving which is why angular momentum is conserved. 
# If you hole the sun stationary angular momentum will not be converved 
# The angular momentum is constant for the system 