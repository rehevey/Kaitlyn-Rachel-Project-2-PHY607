from vpython import *

g1 = graph(xtitle="t", ytitle="p", width=400, height=200)
fap = gcurve(colot=color.red)
fbp = gcurve(color=color.blue) 
ftp = gcurve(color=color.purple) 

g2 = graph(xtitle="t", ytitle="KE", width=400, height=200)
faK = gcurve(colot=color.red)
fbK = gcurve(color=color.blue) 
ftK = gcurve(color=color.purple) 

a = sphere(pos=vector(-1,0.05,0), radius=0.1, color=color.yellow, make_trail=True)
b = sphere(pos=vector(0,0,0), radius=0.01, color=color.cyan, make_trail=True) 

a.m = 0.2
b.m = 0.1
b.p = b.m*vector(0,0,0)
a.p = a.m * vector(0.3, 0,0)    # a is moving towards b 

c = 0.0003 #force coefficent 
t = 0
dt = 0.001

while t<5:
    rate(1000)
    r = b.pos - a.pos 
    FAB = c*norm(r)/mag(r)**3 
    b.p = b.p + FAB*dt 
    a.p = a.p - FAB*dt 
    b.pos = b.pos + b.p*dt/b.m
    a.pos = a.pos + a.p*dt/a.m 
    t = t + dt 
    Ka = mag(a.p)**2 / (2*a.m)
    Kb = mag(b.p)**2 / (2*b.m)

    fap.plot(t, a.p.x)
    fbp.plot(t, b.p.x)
    ftp.plot(t, a.p.x + b.p.x)

    faK.plot(t, Ka)
    fbK.plot(t, Kb)
    ftK.plot(t, Ka + Kb)

print('Simulation finished')
while True:
    rate(10)