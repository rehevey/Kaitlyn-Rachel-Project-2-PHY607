import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen

# ----- Rosenbrock Graph -----

# Create a grid of points in the x-y plane
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)

# Initialize an array to store Rosenbrock function values
Z = np.zeros_like(X)

# Calculate the Rosenbrock function for each point on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosen([X[i, j], Y[i, j]])

# Create the plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap='plasma')
plt.title('Rosenbrock Function (using scipy.optimize.rosen)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='f(x, y)')
plt.show()

#Equations
def func(p):
    x, y = p 
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 

Corners = [(-2, -1), (-2, 3), (2, -1), (2, 3)]  

# ----- X and Y range -----

Xrange = (-2, 2)
Yrange = (-1, 3)

# ----- Estimating Slope (Finite Differences) -----

def slope (p, h):
    h = 0.000001
    x, y = p
    dfdx = (func((x + h, y)) - func((x - h, y))) / (2*h)
    dfdy = (func((x, y + h)) - func((x, y - h))) / (2*h)
    return np.array([dfdx, dfdy])

# Hessian a 2x2 matrix holding the local curvature 

def hessian (p ,h):
    p = np.array(p, dtype=float)     # so that p + step works for tuples too
    h = 0.0001
    H = np.zeros((2,2)) # make the array aka the matrix 
    for i in range(2): # loop twice, i = 0 for x direction and i = 1 for y direction
        step = np.zeros(2)
        step[i] = h
        H[:, i] = (slope(p + step) - slope(p - step)) / (2 * h) # stores the move in x and y (":" for all rows, "i" is the column), first pass fills in x and second fills in y 
    return H

# ----- Brute Force / Explicit Exploration of the Grid -----

def brute_force(n):
    xlist = np.linspace(-2, 2, n)
    ylist = np.linspace(-1, 3, n) 
    count = 0
    best = np.inf       # starter value for best (positive infinity)
    best_points = None  # value starts empty 
    for xval in xlist:
        for yval in ylist:
            count += 1 
            val = func(np.array([xval, yval]))
            if val < best:
                best = val                # replace best with the new best value 
                best_points = (xval, yval)
    return best_points, best, count 

best_points, best, count = brute_force(400)
print('best points:', best_points, ' best value:', best, ' count:', count)

# ----- Gradient Descent (gradient from finite differences) -----

def gradient_descent (start_x, start_y, moving_rate, max_runs=2000000, tol=1e-10):
    xg = start_x
    yg = start_y
    history = []

    # Gradient Descent Iterations 
    for i in range(1,max_runs+1):
        grad_x, grad_y = slope((xg, yg), 0.000001)

        xg = xg - moving_rate * grad_x
        yg = yg - moving_rate * grad_y
        val = func((xg, yg))

        history.append((xg, yg, val))

        if val < tol:                          # close enough: stop and report
            return xg, yg, val, history, i
        if not np.isfinite(val) or val > 1e20: # blew up: step size too large
            return xg, yg, val, history, None

    return xg, yg, val, history, None          # never got there in max_runs

print(gradient_descent(-2, -1, 1e-3)[-1])     

# ----- Newton's Method -----



# ----- Nelder-Mead ----- (Tuesday 10/20)

