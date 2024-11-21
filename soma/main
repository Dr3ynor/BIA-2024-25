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
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

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

# Add other functions if needed

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

def run_soma(benchmark_function, bounds):
    # SOMA Parameters
    num_particles = 30
    dimensions = 2
    iterations = 50
    prt = 0.4
    step = 0.11
    path_length = 3

    # Initialize particles
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
    fitness = [benchmark_function(pos) for pos in positions]
    leader = positions[np.argmin(fitness)]
    leader_fitness = np.min(fitness)

    # Record positions for animation
    results = [positions.copy()]

    # Function to perform one SOMA iteration
    def soma_step():
        nonlocal positions, leader, leader_fitness
        new_positions = positions.copy()
        for i in range(num_particles):
            if np.array_equal(leader, positions[i]):
                continue
            best_pos = positions[i]
            for t in np.arange(0, path_length + step, step):
                r = np.random.uniform(0, 1, dimensions) < prt
                candidate = positions[i] + t * r * (leader - positions[i])
                candidate = np.clip(candidate, bounds[0], bounds[1])
                if benchmark_function(candidate) < benchmark_function(best_pos):
                    best_pos = candidate
            new_positions[i] = best_pos
        positions = new_positions
        fitness[:] = [benchmark_function(pos) for pos in positions]
        leader = positions[np.argmin(fitness)]
        leader_fitness = np.min(fitness)
        results.append(positions.copy())

    # Create grid for surface plot
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([benchmark_function([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

    # Plotting setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7, cmap='viridis', edgecolor='none')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(Z.min(), Z.max())

    particles, = ax.plot([], [], [], 'ro', markersize=5)

    # Update function for animation
    def animate(frame):
        soma_step()
        current_positions = results[frame]
        particles.set_data(current_positions[:, 0], current_positions[:, 1])
        particles.set_3d_properties([benchmark_function(pos) for pos in current_positions])
        return particles,

    ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=200, blit=False)
    plt.show()

# Main loop
while True:
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

    benchmark_function, bounds = benchmark_functions[selected_function_name]
    run_soma(benchmark_function, bounds)
