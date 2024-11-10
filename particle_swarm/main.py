import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Define benchmark functions
def sphere(x):
    return np.sum(x**2)

def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

# PSO Parameters
num_particles = 30
dimensions = 2
iterations = 50
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 2.0
bounds = (-10, 10)

# Initialize particles
positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
personal_best_positions = np.copy(positions)
personal_best_scores = np.array([sphere(pos) for pos in positions])  # Change to desired function
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# Function to update particle positions
def update_particles():
    global positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score

    for i in range(num_particles):
        # Calculate new velocity
        inertia = inertia_weight * velocities[i]
        cognitive = cognitive_coeff * np.random.rand() * (personal_best_positions[i] - positions[i])
        social = social_coeff * np.random.rand() * (global_best_position - positions[i])
        velocities[i] = inertia + cognitive + social

        # Update position and apply bounds
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        # Evaluate the function
        score = sphere(positions[i])  # Change to desired function

        # Update personal best
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

    # Update global best
    best_particle = np.argmin(personal_best_scores)
    if personal_best_scores[best_particle] < global_best_score:
        global_best_score = personal_best_scores[best_particle]
        global_best_position = personal_best_positions[best_particle]

# Define grid for plotting the function surface
x = np.linspace(bounds[0], bounds[1], 100)
y = np.linspace(bounds[0], bounds[1], 100)
X, Y = np.meshgrid(x, y)

# Function to calculate Z values for surface plot (change `sphere` to desired function)
Z = np.array([sphere(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())])
Z = Z.reshape(X.shape)

# Plotting setup with surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[0], bounds[1])
ax.set_zlim(0, np.max(Z) * 1.1)  # Adjust based on function scale

# Plot the function surface
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none')

particles, = ax.plot([], [], [], 'bo', ms=5)

# Update function for animation
def animate(i):
    update_particles()
    particles.set_data(positions[:, 0], positions[:, 1])
    particles.set_3d_properties([sphere(pos) for pos in positions])  # Change to desired function
    return particles,

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=100, blit=True)
plt.show()