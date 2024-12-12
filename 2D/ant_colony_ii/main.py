import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

NUM_CITIES = 20
NUM_ANTS = NUM_CITIES
PHEROMONE_IMPORTANCE = 1
DISTANCE_IMPORTANCE = 2
NUM_MIGRATIONS = 200
VAPORIZATION_COEFFICIENT = 0.5

def generate_cities(count):
    return [np.random.uniform(low=0, high=200, size=2) for _ in range(count)]


def calculate_euclidean_distance(first, second):
    first_part = first[0] - second[0]
    second_part = first[1] - second[1]
    return np.sqrt((np.power(first_part, 2) + np.power(second_part, 2)))


def calculate_distance_matrix(cities):
    count = len(cities)
    matrix = np.zeros((count, count))
    
    for i in range(count):
        for j in range(i, count):
            if i == j:
                continue
            distance = calculate_euclidean_distance(cities[i], cities[j])
            matrix[i,j] = distance
            matrix[j,i] = distance
    return matrix

def compute_distance(individual, matrix, compute_end_start=True):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += matrix[individual[i], individual[i + 1]]
    if compute_end_start:
        total_distance += matrix[individual[0], individual[len(individual) - 1]]
    return total_distance

def animate_connections(cities, solutions, distances):
    fig, ax = plt.subplots(figsize=(8, 8))
    cities = np.array(cities)
    scatter = ax.scatter(cities[:, 0], cities[:, 1], color='red', zorder=2)

    distance_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='blue')
    generation_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='green')
    path_line, = ax.plot([], [], color='blue', alpha=0.7, linewidth=2, zorder=1)
    
    def update(frame):
        solution = solutions[frame]
        distance = distances[frame]
        
        path = cities[solution]
        path_line.set_data(path[:, 0], path[:, 1])
        
        distance_text.set_text(f'Distance: {distance:.2f}')
        generation_text.set_text(f'Generation: {frame}/{len(solutions)-1}')
        
        return path_line, scatter, distance_text, generation_text
    
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title('Ant Colony Optimization Path Evolution')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')

    anim = FuncAnimation(
        fig, update, frames=len(solutions), interval=100, blit=True
    )
    return anim



def calculate_visibility_matrix(distance_matrix: NDArray[np.float64]):
    shape = distance_matrix.shape[0]
    visibility_matrix = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            if i == j: 
                continue
            value = 1 / distance_matrix[i,j]
            visibility_matrix[i,j] = value
            visibility_matrix[j,i] = value
    return visibility_matrix

def calculate_probabilities(pheromone_matrix, visibility_matrix, current_city):
    arr = []
    for target_city in range(NUM_CITIES):
        pheromone = pheromone_matrix[current_city, target_city] ** PHEROMONE_IMPORTANCE
        distance = visibility_matrix[current_city, target_city] ** DISTANCE_IMPORTANCE
        arr.append(pheromone * distance)
    
    sum_weights = np.sum(arr)
    for i in range(NUM_CITIES):
        if arr[i] <= 0:
            continue
        arr[i] = arr[i] / sum_weights
    return arr

cities = generate_cities(NUM_CITIES)
distance_matrix = calculate_distance_matrix(cities)
visibility_matrix = calculate_visibility_matrix(distance_matrix)
pheromone_matrix = np.ones(distance_matrix.shape)

best_solution = None
best_distance = float('inf')
best_idx = -1

solutions = []
distances = []
start = time.time()
for migration in range(NUM_MIGRATIONS):
    current_solutions = []
    for ant in range(NUM_ANTS):
        current_ant_visibility_matrix = np.copy(visibility_matrix)
        solution = [ant]
        current_city = solution[-1]
        current_ant_visibility_matrix[:, current_city] = 0
        for _ in range(NUM_CITIES - 1):
            probabilities = calculate_probabilities(pheromone_matrix, current_ant_visibility_matrix, current_city)
            r = np.random.uniform()
            cumulative = np.cumsum(probabilities)
            city = np.where((r < cumulative) & (cumulative > r))[0][0]
            solution.append(city)
            current_city = city
            current_ant_visibility_matrix[:, current_city] = 0
        solution.append(ant)
        current_solutions.append(solution)
    
    pheromone_matrix *= VAPORIZATION_COEFFICIENT
    current_distances = []
    for solution in current_solutions:
        current_distances.append(compute_distance(solution, distance_matrix, compute_end_start=False))
        for i in range(len(solution) - 2):
            idx_1 = solution[i]
            idx_2 = solution[i + 1]
            pheromone_matrix[idx_1, idx_2] += 1 / current_distances[-1]

    min_idx = np.argmin(current_distances)
    if current_distances[min_idx] < best_distance:
        best_solution = current_solutions[min_idx]
        best_distance = current_distances[min_idx]
        best_idx = migration
    solutions.append(current_solutions[min_idx])
    distances.append(current_distances[min_idx])

solutions.append(best_solution)
distances.append(best_distance)
print(f"Elapsed: {time.time() - start}")
animation = animate_connections(np.array(cities), solutions, distances)
plt.show()
