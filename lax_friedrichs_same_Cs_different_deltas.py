# Solving the 1D linear advection equation using the Lax-Friedrichs scheme
# For same values of Courant number C
# No animation, just snapshots of t = 0, tf/3, 2tf/3, tf
# Only case 1 initial condition is used here
import numpy as np
import matplotlib.pyplot as plt

# Function to implement the Lax-Friedrichs scheme
# Initial condition case 1: u(x,0) = f(x) = exp(-1/(1-x**2)) for |x| < 1, 0 otherwise
# Domain: x in [-xf, xf], t in [0, tf]
# Boundary conditions: u(-xf,t) = 0, u(xf,t) = 0
# Define x = 0 at the center of the domain
def ftcs_advection(xf, nx, tf, nt, c):
    # Spatial discretisation
    x = np.linspace(-xf, xf, nx)
    dx = x[1] - x[0]
    
    # Temporal discretisation
    dt = tf / nt
    
    # Initialize the solution array
    # u = u(t, x)
    u = np.zeros((nt, nx))
    
    # Initial condition
    # Index variable for spatial position is l
    for l in range(nx):
        # Case 1: f(x)
        # if abs(x[l]) < 1:
        #     u[0, l] = np.exp(-1 / (1 - x[l]**2))
        # else:
        #     u[0, l] = 0.0
        # Case 2: g(x)
        if 0 < x[l] < 1:
            u[0, l] = x[l]
        else:
            u[0, l] = 0.0
    
    # Lax-Friedrichs scheme
    # Index variable for time is n
    for n in range(0, nt - 1):
        for l in range(1, nx - 1):
            u[n + 1, l] = 0.5 * (u[n, l + 1] + u[n, l - 1]) - (c * dt / (2 * dx)) * (u[n, l + 1] - u[n, l - 1])
        
        # Boundary conditions
        u[n + 1, 0] = 0.0
        u[n + 1, -1] = 0.0
    
    return x, u

# Parameters
# Courant numbers constant, vary nx and nt accordingly
# C roughly equal to 0.21 for all cases
xf = 3.0      # Spatial domain limit
nx = [2001, 201, 51]     # Number of spatial points
tf = 1.0      # Final time
nt = [4000, 400, 100]     # Number of time steps
c = 2.5       # Wave speed

# Plotting results for different Courant numbers
# Plot all tf/3, 2tf/3, tf on the same graph for each case
fig2, axs = plt.subplots(1, 3, figsize=(10, 8))
colours = ['b', 'g', 'r']
for j in range(len(nx)):
    x, u = ftcs_advection(xf, nx[j], tf, nt[j], c)
    time_snapshots = [nt[j] // 3, 2 * nt[j] // 3, nt[j] - 1]
    for i, t in enumerate(time_snapshots):
        ax = axs[i]
        ax.plot(x, u[t, :], color=colours[j], label=f'Δx={xf*2/nx[j]:.3f}, Δt={tf/nt[j]:.4f}, Max={u[t, np.argmax(u[t, :])]:.2f}, Min={u[t, np.argmin(u[t, :])]:.2f}') # Put max and min points in label
        ax.plot(x[np.argmax(u[t, :])], u[t, np.argmax(u[t, :])], f'{colours[j]}o')  # Max point
        ax.plot(x[np.argmin(u[t, :])], u[t, np.argmin(u[t, :])], f'{colours[j]}o')  # Min point
        # ax.text(x[np.argmax(u[t, :])], u[t, np.argmax(u[t, :])] + 0.10,
        #         f'Max: {u[t, np.argmax(u[t, :])]:.2f}', fontsize=10, color=f'{colours[j]}', ha='center') # Max label
        # ax.text(x[np.argmin(u[t, :])], u[t, np.argmin(u[t, :])] - 0.10,
        #         f'Min: {u[t, np.argmin(u[t, :])]:.2f}', fontsize=10, color=f'{colours[j]}', ha='center') # Min label
        ax.set_title(f't = {t * (tf / nt[j]):.2f} s')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_xlim(-xf, xf)
        ax.set_ylim(-0.05, 0.85)
        ax.legend(fontsize=12, loc='upper left')
    fig2.suptitle('1D Linear Advection using Lax-Friedrichs Scheme, Initial Condition g(x) - Same Courant Numbers')
plt.tight_layout()
plt.subplots_adjust(left=0.048, bottom=0.62, right=0.985, top=0.913, wspace=0.143, hspace=0.202)
plt.show()