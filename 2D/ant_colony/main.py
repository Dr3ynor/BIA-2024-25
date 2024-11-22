import numpy as np
import matplotlib.pyplot as plt

NUM_CITIES = 50
NUM_ITERATIONS = 300
NUM_ANTS = 10
# Funkce pro výpočet vzdáleností mezi městy
def calculate_distances(cities):
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i, n_cities):
            distances[i, j] = distances[j, i] = np.linalg.norm(cities[i] - cities[j])
    return distances

# ACO implementace
class AntColonyOptimizer:
    def __init__(self, cities, n_ants, n_iterations, alpha=1, beta=2, evaporation_rate=0.5, pheromone_init=0.1):
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.distances = calculate_distances(cities)
        self.pheromones = np.full((self.n_cities, self.n_cities), pheromone_init)
        self.best_path = None
        self.best_distance = np.inf
        self.best_routes_history = []  # Uchovává historii nejlepších tras

    def run(self):
        for iteration in range(self.n_iterations):
            all_paths = []
            all_distances = []
            for _ in range(self.n_ants):
                path, distance = self.construct_solution()
                all_paths.append(path)
                all_distances.append(distance)

                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path

            self.best_routes_history.append((self.best_path, self.best_distance))
            self.update_pheromones(all_paths, all_distances)
            print(f"Iteration {iteration + 1}: Best distance = {self.best_distance:.2f}")

        return self.best_path, self.best_distance

    def construct_solution(self):
        path = [np.random.randint(self.n_cities)]
        while len(path) < self.n_cities:
            current_city = path[-1]
            probabilities = self.transition_probabilities(current_city, path)
            next_city = np.random.choice(range(self.n_cities), p=probabilities)
            path.append(next_city)
        distance = self.calculate_path_distance(path)
        return path, distance

    def transition_probabilities(self, current_city, visited):
        probabilities = np.zeros(self.n_cities)
        for i in range(self.n_cities):
            if i not in visited:
                probabilities[i] = (self.pheromones[current_city, i] ** self.alpha) * \
                                   ((1 / self.distances[current_city, i]) ** self.beta)
        probabilities /= probabilities.sum()
        return probabilities

    def calculate_path_distance(self, path):
        distance = sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))
        distance += self.distances[path[-1], path[0]]  # Návrat k počátečnímu městu
        return distance

    def update_pheromones(self, all_paths, all_distances):
        self.pheromones *= (1 - self.evaporation_rate)
        for path, distance in zip(all_paths, all_distances):
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += 1 / distance
            self.pheromones[path[-1], path[0]] += 1 / distance  # Návratová hrana

# Vizualizace průběhu algoritmu
def visualize_progress(cities, best_routes_history):
    plt.figure(figsize=(8, 6))
    x = cities[:, 0]
    y = cities[:, 1]
    plt.scatter(x, y, c='red', s=50, label="Cities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    line, = plt.plot([], [], 'b-', lw=2)  # Inicializace čáry
    input("Press Enter...")
    for i, (route, distance) in enumerate(best_routes_history):
        plt.title(f"Ant Colony Optimization Progress {i + 1}/{len(best_routes_history)} | Distance: {distance:.2f}")
        # print(f"Iteration {i + 1}: Route {route}, Distance: {distance:.2f}")
        route_x = np.append(cities[route, 0], cities[route[0], 0])
        route_y = np.append(cities[route, 1], cities[route[0], 1])
        line.set_xdata(route_x)
        line.set_ydata(route_y)
        plt.pause(0.1)

    plt.show()  # Zachová graf po skončení


# Příklad použití
np.random.seed(42)
cities = np.random.rand(NUM_CITIES, 2) * 100  # 10 měst náhodně umístěných v ploše 100x100
aco = AntColonyOptimizer(cities, n_ants=NUM_ANTS, n_iterations=NUM_ITERATIONS, alpha=1, beta=2, evaporation_rate=0.1)
best_path, best_distance = aco.run()

# Vizualizace průběhu
visualize_progress(cities, aco.best_routes_history)
