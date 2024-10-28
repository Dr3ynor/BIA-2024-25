import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
NP = 25            # Moderate population size for exploration
G = 20_000            # Sufficient generations for convergence
D = 20             # Fixed number of cities
MUTATION_RATE = 0.5  # 50% mutation probability
RANGE = 200
# Generate random cities (coordinates in a 2D plane)
cities = np.random.rand(D, 2) * RANGE
starting_city = 0  # Fixed starting city (e.g., the first city)

# Function to calculate the total distance of a route
def calculate_distance(route):
    # Include distance from last city back to the starting city
    total_distance = sum(np.linalg.norm(cities[route[i]] - cities[route[i + 1]]) for i in range(-1, D - 1))
    total_distance += np.linalg.norm(cities[route[-1]] - cities[route[0]])  # Return to start
    return total_distance

# Generate initial population of random routes
def generate_population():
    population = []
    for _ in range(NP):
        route = [starting_city] + random.sample(range(1, D), D - 1)  # Keep starting city fixed
        population.append(route)
    return population

# Order crossover function
def crossover(parent_A, parent_B):
    start, end = sorted(random.sample(range(1, D), 2))  # Only shuffle non-starting cities
    offspring = [None] * D
    offspring[0] = starting_city  # Fixed starting city
    offspring[start:end] = parent_A[start:end]
    pos = end
    for city in parent_B:
        if city not in offspring:
            offspring[pos % D] = city
            pos += 1
    return offspring

# Swap mutation function
def mutate(route):
    a, b = random.sample(range(1, D), 2)  # Only swap non-starting cities
    route[a], route[b] = route[b], route[a]
    return route

# Evaluate the population and return sorted population by fitness (distance)
def evaluate_population(population):
    population = sorted(population, key=calculate_distance)
    return population

# Main genetic algorithm function
def genetic_algorithm():
    population = generate_population()
    best_route = None
    best_distance = float("inf")

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    scatter = ax.scatter(cities[:, 0], cities[:, 1], color="red")
    line, = ax.plot([], [], "b-", linewidth=1)
    plt.title("Genetic Algorithm TSP")

    for generation in range(G):
        new_population = []

        for i in range(NP):
            parent_A = population[i]
            parent_B = random.choice([p for p in population if p != parent_A])

            offspring = crossover(parent_A, parent_B)

            # Mutate with a probability of MUTATION RATE (%)
            if random.random() < MUTATION_RATE:
                offspring = mutate(offspring)

            # Replace if offspring is better
            if calculate_distance(offspring) < calculate_distance(parent_A):
                new_population.append(offspring)
            else:
                new_population.append(parent_A)

        population = evaluate_population(new_population)

        # Track the best route
        current_best_distance = calculate_distance(population[0])
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = population[0]

        # Visualization update
        route_coords = cities[best_route + [best_route[0]]]  # Return to start
        line.set_data(route_coords[:, 0], route_coords[:, 1])
        ax.set_title(f"Generation: {generation + 1}, Distance: {best_distance:.2f}")
        plt.draw()
        plt.pause(0.0000000000001)

    plt.ioff()  # Turn off interactive mode
    plt.show()
    return best_route, best_distance

# Run the genetic algorithm and plot the best solution
best_route, best_distance = genetic_algorithm()
print("Best route:", best_route)
print("Best distance:", best_distance)
