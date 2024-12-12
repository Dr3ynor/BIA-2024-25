import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


num_individuals= 50
dim = 30
num_generations = 100


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


def teacher_learner_algorithm(func, bounds):
    num_of_iterations = 0
    MAX_ITERATIONS = 3000
    population = np.random.uniform(bounds[0], bounds[1], (num_individuals, dim))
    fitness = np.array([func(ind) for ind in population])
    num_of_iterations = num_of_iterations + num_individuals
    best_value = np.min(fitness)
    best_position = population[np.argmin(fitness)]
    all_positions = [population.copy()]

    for _ in range(num_generations):
        # Fáze učitele
        teacher = population[np.argmin(fitness)] # Nejlepší jedinec v populaci se stane učitelem
        mean = np.mean(population, axis=0) # Vypočítá se průměrná pozice všech jedinců v populaci (všech studentů)
        TF = np.random.choice([1, 2]) # Náhodně se vybere faktor TF (1 nebo 2) - TF je váha, která určuje, jak moc učitel ovlivní studenty 1 - menší vliv, 2 - větší vliv
        new_population = population + np.random.uniform(size=(num_individuals, dim)) * (teacher - TF * mean) # Update pozice studentů na základě učitele podle vzorce

        new_population = np.clip(new_population, bounds[0], bounds[1]) # Clipnutí hodnot, které přesahují hranice definičního oboru

        new_fitness = np.array([func(ind) for ind in new_population]) # Ohodnocení nových pozic
        num_of_iterations = num_of_iterations + new_population.shape[0]
        if num_of_iterations >= MAX_ITERATIONS:
            return best_position, best_value, all_positions
        improved = new_fitness < fitness # Zjistíme, které nové pozice jsou lepší než ty původní (studenti, kteří se zlepšili)
        population[improved] = new_population[improved] # Pokud se student zlepšil, jeho pozice se změní
        fitness[improved] = new_fitness[improved] # Pokud se student zlepšil, jeho fitness se změní

        # Fáze učení
        for i in range(num_individuals): # Pro každého studenta v populaci se provede učení s jiným studentem (partnerem)
            partner = np.random.choice([j for j in range(num_individuals) if j != i]) # Náhodný výběr studenta (partnera) k učení (kromě sebe samotného)
            # if else podmínka pro výpočet nové pozice studenta na základě partnera podle vzorce (pokud je partner lepší než student, tak se student přesune k partnerovi, jinak se partner přesune k studentovi)
            if fitness[i] < fitness[partner]:
                new_ind = population[i] + np.random.uniform(size=dim) * (population[i] - population[partner])
            else:
                new_ind = population[i] + np.random.uniform(size=dim) * (population[partner] - population[i])
            new_ind = np.clip(new_ind, bounds[0], bounds[1])
            new_fitness = func(new_ind)
            num_of_iterations = num_of_iterations + 1
            if num_of_iterations >= MAX_ITERATIONS:
                return best_position, best_value, all_positions
            if new_fitness < fitness[i]:
                population[i] = new_ind
                fitness[i] = new_fitness


        current_best_value = np.min(fitness) # Uložení nejlepšího jedince
        current_best_position = population[np.argmin(fitness)] # Uložení nejlepší pozice
        if current_best_value < best_value: # Pokud je nový nejlepší jedinec lepší než ten předchozí, tak se aktualizuje nejlepší hodnota a pozice
            best_value = current_best_value
            best_position = current_best_position

        all_positions.append(population.copy())

    return best_position, best_value, all_positions


def visualize_teacher_learner(func, bounds, all_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    def update(frame):
        ax.cla()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        population = all_positions[frame]
        mean = np.array([func(ind) for ind in population])
        ax.scatter(population[:, 0], population[:, 1], mean, color='red')
        ax.set_title(f"Generation {frame + 1}")

    ani = animation.FuncAnimation(fig, update, frames=len(all_positions),interval=100, repeat=False)
    plt.show()


def run_teacher_learner(func,bounds):
    best_position, best_value, all_positions = teacher_learner_algorithm(func, bounds)
    print(f"Best Position: {best_position}\nBest Value: {best_value}")
    input("Press Enter...")
    visualize_teacher_learner(func, bounds, all_positions)



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
    # run_teacher_learner(benchmark_function, bounds)

    for i in range(30):
        best_position, best_value, all_positions = teacher_learner_algorithm(benchmark_function, bounds)
        print(f"{best_value}")
