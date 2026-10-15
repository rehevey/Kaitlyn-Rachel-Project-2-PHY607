import numpy as np 
import time 
import matplotlib.pyplot as plt

def plot_construction():
    fig, axis = plt.subplots(1, 2, figsize=(10, 5))

    #Generated Dots 
    axis[0].set_title('Random Gnerated Dots')
    axis[0].set_aspect('equal')
    axis[0].set_xlim([-1, 1])
    axis[0].set_ylim([-1, 1])

    #Plot of Approximation 
    axis[1].set_title('Approximating pi')
    axis[1].set_xlim([0, 1000])
    axis[1].set_ylim([0, 10])
    axis[1].grid()

    return axis

axis = plot_construction()

N_samples = np.arange(1, 1000)
pi_counter = 0 

dots_x = []
dots_y = []

dots_color = []
pi_array = []


for N in N_samples:

    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)

    dots_x.append(x)
    dots_y.append(y)

    if np.sqrt(x**2 + y**2) <= 1:
        pi_counter += 1
        dots_color.append('red')

    else:
        dots_color.append('blue')

    proability = pi_counter/N
    approx_pi = 4*proability

    pi_array.append(approx_pi)

    axis[0].scatter(dots_x, dots_y, color=dots_color, marker='o', s=5)
    axis[0].set_title(f"Red Dots: {pi_counter}, Blue Dots: {N-pi_counter}")

    axis[1].plot(range(1, N+1), pi_array, 'black')
    axis[1].set_title(f'Approximating pi {approx_pi: .5f}')

    plt.draw()
    plt.pause(0.01)

    # print(
    #     f"For {N}-samples, Approximated Pi = {approx_pi: .5f},"
    #     f"Error: {100 * np.abs(approx_pi - np.pi)/np.pi:.3f}%", end='/r')

    # time.sleep(0.001)

plt.show() 