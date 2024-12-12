import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Define benchmark functions
def sphere(params):
    return sum(p**2 for p in params)

def ackley(params, a=20, b=0.2, c=2 * np.pi):
    params = np.array(params)
    d = len(params)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(params**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * params)) / d)
    return term1 + term2 + a + np.exp(1)

def rastrigin(params):
    return sum(p**2 - 10 * np.cos(2 * np.pi * p) + 10 for p in params)

def rosenbrock(params):
    x = params[0]
    y = params[1]
    return (1-x)**2 + 100*(y-x**2)**2

def griewank(params):
    sum_part = sum(p**2 for p in params) / 4000
    prod_part = np.prod([np.cos(p / np.sqrt(i + 1)) for i, p in enumerate(params)])
    return sum_part - prod_part + 1

def schwefel(params):
    return 418.9829 * len(params) - sum(p * np.sin(np.sqrt(abs(p))) for p in params)

def levy(params):
    w = 1 + (np.array(params) - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz(params):
    sum_part = 0
    d = len(params)
    for i in range(d):
        xi = params[i]
        sum_part += np.sin(xi) * (np.sin(((i + 1) * xi**2) / np.pi))**(2 * 10)
    return -sum_part

def zakharov(params):
    sum1 = sum(p**2 for p in params)
    sum2 = sum(0.5 * (i + 1) * p for i, p in enumerate(params))
    return sum1 + sum2**2 + sum2**4

benchmark_functions = {
    "1": (sphere, [-5.12, 5.12]),
    "2": (ackley, [-30, 30]),
    "3": (rastrigin, [-5.12, 5.12]),
    "4": (rosenbrock, [-2, 2]),
    "5": (griewank, [-5, 10]),
    "6": (schwefel, [-500, 500]),
    "7": (levy, [-10, 10]),
    "8": (michalewicz, [0, 3.1415926535]),
    "9": (zakharov, [-5, 5])
}


def particle_swarm(benchmark_function, bounds):
    MAX_EVALUATIONS = 3000
    num_of_evaluations = 0
    # PSO Parametry
    num_particles = 30
    dimensions = 30
    iterations = 50
    inertia_weight = 0.5 # setrvačnost
    cognitive_coeff = 1.5 # k nejlepší svoji pozici
    social_coeff = 2.0 # k nejlepší pozici celé skupiny

    # Počáteční částice
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
    velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.array([benchmark_function(pos) for pos in positions])
    num_of_evaluations += num_particles
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # nonlocal positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score
    for i in range(num_particles):
        inertia = inertia_weight * velocities[i]
        cognitive = cognitive_coeff * np.random.rand() * (personal_best_positions[i] - positions[i])
        social = social_coeff * np.random.rand() * (global_best_position - positions[i])
        velocities[i] = inertia + cognitive + social
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])
        score = benchmark_function(positions[i])
        num_of_evaluations = num_of_evaluations + 1
        if num_of_evaluations >= MAX_EVALUATIONS:
            return global_best_score
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]
    best_particle = np.argmin(personal_best_scores)
    if personal_best_scores[best_particle] < global_best_score:
        global_best_score = personal_best_scores[best_particle]
        global_best_position = personal_best_positions[best_particle]
    return global_best_score
"""
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([benchmark_function([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    # For Michalewicz, dynamically adjust zlim to zoom in on the important range
    if benchmark_function == michalewicz:
        ax.set_zlim(-1, 1)
    else:
        ax.set_zlim(0, np.max(Z) * 1.1)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7, cmap='viridis', edgecolor='none')

    particles, = ax.plot([], [], [], 'bo', ms=5)

    # Update function for animation
    def animate(i):
        particle_swarm()
        particles.set_data(positions[:, 0], positions[:, 1])
        particles.set_3d_properties([benchmark_function(pos) for pos in positions])
        return particles,

    # Run the animation
    ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=100, blit=True)
    plt.show()
"""
# Main loop
while True:
    # Prompt the user to select a function
    print("\nSelect a benchmark function:")
    for key, (func, _) in benchmark_functions.items():
        print(f"{key}: {func.__name__}")
    print("0: Exit")

    selected_function_name = input("Enter a number (1-9 or 0 to exit): ").strip()
    
    if selected_function_name == "0":
        print("Exiting.")
        break
    elif selected_function_name not in benchmark_functions:
        print("Invalid selection. Try again.")
        continue

    # Get the selected function and its range
    benchmark_function, bounds = benchmark_functions[selected_function_name]
    
    # Run PSO with the selected function
    for i in range(30):
        print(particle_swarm(benchmark_function, bounds))
