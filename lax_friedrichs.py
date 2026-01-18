# Solving the 1D linear advection equation using Lax-Friedrichs scheme
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to implement the Lax-Friedrichs scheme
# Initial condition case 1: u(x,0) = f(x) = exp(-1/(1-x**2)) for |x| < 1, 0 otherwise
# Initial condition case 2: u(x,0) = g(x) = x for 0 < x < 1, 0 otherwise
# Domain: x in [-xf, xf], t in [0, tf]
# Boundary conditions: u(-xf,t) = 0, u(xf,t) = 0
# Define x = 0 at the center of the domain
def lax_friedrichs_advection(xf, nx, tf, nt, c):
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
        if abs(x[l]) < 1:
            u[0, l] = np.exp(-1 / (1 - x[l]**2))
        else:
            u[0, l] = 0.0
        # Case 2: g(x)
        # if 0 < x[l] < 1:
        #     u[0, l] = x[l]
        # else:
        #     u[0, l] = 0.0
    
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
xf = 3.0      # Spatial domain limit
nx = 101      # Number of spatial points
tf = 1.0      # Final time
nt = 200      # Number of time steps
c = 2.5       # Wave speed

# Run the Lax-Friedrichs advection solver
x, u = lax_friedrichs_advection(xf, nx, tf, nt, c)

# Animation of the results
# Also make a mark of the maximum and minimum values of u at each time step, with text labels
fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :], color='b')
max_point, = ax.plot([], [], 'ro')  # Point for maximum
min_point, = ax.plot([], [], 'go')  # Point for minimum
max_text = ax.text(0, 0, '', fontsize=10, color='red', ha='center')
min_text = ax.text(0, 0, '', fontsize=10, color='green', ha='center')
ax.set_xlim(-xf, xf)
ax.set_ylim(-0.05, 0.85) # Case 1
# ax.set_ylim(-0.2, 1.2) # Case 2
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('1D Linear Advection using Lax-Friedrichs Scheme')
def update(frame):
    # Update the main line
    line.set_ydata(u[frame, :])
    # Update max and min points
    max_idx = np.argmax(u[frame, :])
    min_idx = np.argmin(u[frame, :])
    max_point.set_data([x[max_idx]], [u[frame, max_idx]])
    min_point.set_data([x[min_idx]], [u[frame, min_idx]])
    # Update text labels
    max_text.set_position((x[max_idx], u[frame, max_idx] + 0.10))
    max_text.set_text(f'Max: {u[frame, max_idx]:.2f}')
    min_text.set_position((x[min_idx], u[frame, min_idx] - 0.10))
    min_text.set_text(f'Min: {u[frame, min_idx]:.2f}')
    ax.set_title(f'1D Linear Advection using Lax-Friedrichs Scheme\nt = {frame * (tf / nt):.2f} s')
    return line, max_point, min_point, max_text, min_text
ani = animation.FuncAnimation(fig, update, frames=nt, blit=False, interval=50)
plt.show()


# Plot specific time snapshots along with the max and min values
# Make 4 subplots on one figure
# Choose to plot at t = 0, tf/3, 2tf/3, tf
time_snapshots = [0, nt // 3, 2 * nt // 3, nt - 1]
fig2, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, t in enumerate(time_snapshots):
    ax = axs[i // 2, i % 2]
    ax.plot(x, u[t, :], color='b')
    ax.plot(x[np.argmax(u[t, :])], u[t, np.argmax(u[t, :])], 'ro')  # Max point
    ax.plot(x[np.argmin(u[t, :])], u[t, np.argmin(u[t, :])], 'go')  # Min point
    ax.text(x[np.argmax(u[t, :])], u[t, np.argmax(u[t, :])] + 0.10,
            f'Max: {u[t, np.argmax(u[t, :])]:.2f}', fontsize=14, color='red', ha='center') # Max label
    ax.text(x[np.argmin(u[t, :])], u[t, np.argmin(u[t, :])] - 0.10,
            f'Min: {u[t, np.argmin(u[t, :])]:.2f}', fontsize=14, color='green', ha='center') # Min label
    ax.set_title(f't = {t * (tf / nt):.2f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_xlim(-xf, xf)
    ax.set_ylim(-0.05, 0.85) # Case 1
    # ax.set_ylim(-0.2, 1.2) # Case 2
fig2.suptitle('1D Linear Advection using Lax-Friedrichs Scheme, Initial Condition f(x) - Snapshots at Different Times')
plt.tight_layout()
plt.show()