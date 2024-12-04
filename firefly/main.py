import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

LIGHT_ABSORPTION_COEFFICIENT = 2
NUM_GENERATIONS = 50
NUM_FIREFLIES = 50
DIMENSIONS = 2
ATTRACTIVNESS = 1
RANDOMNESS_SCALING_PARAMETER = 0.3

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


def calculate_light_intensity(value: float, firefly, target_firefly):
    return value * np.pow(np.e, -LIGHT_ABSORPTION_COEFFICIENT * np.linalg.norm(firefly - target_firefly))

def calculate_attractiveness(firefly, target_firefly):
    return ATTRACTIVNESS / (1 + np.linalg.norm(firefly - target_firefly))

def move_towards_target(firefly, target_firefly):
    firefly += calculate_attractiveness(firefly, target_firefly) * (target_firefly - firefly)
    firefly += RANDOMNESS_SCALING_PARAMETER * np.random.normal(size=2)


def firefly_algorithm(func, bounds, num_fireflies=NUM_FIREFLIES, num_generations=NUM_GENERATIONS):
    dim = len(bounds)
    population = np.random.uniform(bounds[0], bounds[1], (num_fireflies, dim))
    best_solution = None
    best_value = float('inf')
    all_solutions = []
    all_values = []
    for _ in range(NUM_GENERATIONS):
        current_solutions = []
        current_values = []
        for current_firefly in population:
            value = func(current_firefly)
            for target_firefly in population:
                if current_firefly is target_firefly:
                    continue

                current_firefly_light_intensity = calculate_light_intensity(value, current_firefly, target_firefly)
                target_firefly_light_intensity = calculate_light_intensity(func(target_firefly), current_firefly, target_firefly)
                if target_firefly_light_intensity < current_firefly_light_intensity:
                    move_towards_target(current_firefly, target_firefly)
                value = func(current_firefly)
            current_values.append(value)
            current_solutions.append(np.copy(current_firefly))
        
        all_solutions.append(np.vstack(current_solutions))
        all_values.append(current_values)
        min_idx = np.argmin(current_values)
        if current_values[min_idx] < best_value:
            best_value = current_values[min_idx]
            best_solution = current_solutions[min_idx]
    return best_solution, best_value, all_solutions

def visualize_firefly(func, bounds, all_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    firefly_positions = all_positions[0]
    scatter = ax.scatter(firefly_positions[:, 0], firefly_positions[:, 1],
                         [func(pos) for pos in firefly_positions], c='red', s=30)
    def update(frame):
        firefly_positions = all_positions[frame]
        
        scatter._offsets3d = (
            firefly_positions[:, 0],
            firefly_positions[:, 1],
            [func(pos) for pos in firefly_positions]
        )

        
        return scatter

    ani = animation.FuncAnimation(fig, update, frames=len(all_positions), interval=50, blit=False)
    plt.show()

def run_firefly(func, bounds):
    best_position, best_value, all_positions = firefly_algorithm(func, bounds)
    print(f"Best Position: {best_position}\nBest Value: {best_value}")
    input("Press Enter...")
    visualize_firefly(func, bounds, all_positions)

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
    run_firefly(benchmark_function, bounds)