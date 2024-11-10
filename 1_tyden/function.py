import numpy as np
import plotly.graph_objects as go

class Function:
    def __init__(self,name):
        self.name = name
        print(f"Function: {self.name}")

    # Zobrazení gridu (bez funkce a bez vyhodnocení)
    def init_grid(self,precision,range):
        x = np.linspace(range[0], range[1],precision)
        y = np.linspace(range[0], range[1], precision)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        return x, y, z
    
    # Vyhodnocení gridu
    def evaluate_grid(self, x, y, z, func):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = func([x[i, j], y[i, j]])
        return z
    
    # Vykreslení funkce, pokud jsou zadány nejlepší parametry a hodnoty, tak je zobrazí
    def plot_function(self, x, y, z, best_params=None, best_values=None):
        fig = go.Figure()
        # Vykreslí povrch funkce
        fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale='Viridis', opacity=0.9))
        
        # Pokud jsou funkci předány i body, tak je zobrazí
        if best_params is not None and best_values is not None:
            best_params = np.array(best_params)
            
            # Vykreslí červeně "historicky nejlepší" body, modře nejlepší bod 
            for i in range(len(best_values)):
                z_val = best_values[i]
                if i == len(best_values)-1:
                    fig.add_trace(go.Scatter3d(
                        x=[best_params[i, 0]], 
                        y=[best_params[i, 1]], 
                        z=[z_val], 
                        mode='markers',
                        marker=dict(size=10, color='cyan'),
                        name='Best Point' if i == 0 else ""
                    ))
                else:
                    fig.add_trace(go.Scatter3d(
                    x=[best_params[i, 0]], 
                    y=[best_params[i, 1]], 
                    z=[z_val], 
                    mode='markers',
                    marker=dict(size=7, color='red'),
                    name='Search Points' if i == 0 else ""
                ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title=f'{self.name} Function Value'
            ),
            title=f'{self.name} Function Plot'
        )
        fig.show()


    # (self, search_range, func, alpha=0.9, T_0=1000, T_min=1e-6):
    def differential_evolution(self, func, dimension, lower_bound, upper_bound, population_size=50, generations=1000, F=0.8, CR=0.9):
        # Initialize the population
        pop = [np.random.uniform(lower_bound, upper_bound, dimension) for _ in range(population_size)]
        best_solutions = []
        best_values = []

        # Evaluate initial population
        fitness = [func(ind) for ind in pop]
        best_idx = np.argmin(fitness)
        best_solutions.append(pop[best_idx])
        best_values.append(fitness[best_idx])

        # Main DE loop
        for g in range(generations):
            new_pop = []

            for i in range(population_size):
                # Mutation: Select r1, r2, r3 (distinct indices)
                indices = [idx for idx in range(population_size) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lower_bound, upper_bound)

                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(dimension)])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                if trial_fitness <= fitness[i]:  # Accept if trial is better or equal
                    new_pop.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_pop.append(pop[i])  # Keep original vector

            pop = new_pop

            # Find best solution in this generation
            best_idx = np.argmin(fitness)
            best_solutions.append(pop[best_idx])
            best_values.append(fitness[best_idx])

            # Optionally print or log progress here

        # Return best solutions and values over generations

        print(f"Best value: {best_values[-1]}\nBest parameters: {best_solutions[-1]}\n")
        return best_solutions, best_values





    def generate_initial_solution(self, search_range):
        return np.random.uniform(search_range[0], search_range[1], 2)

    def generate_neighbor(self, params, search_range):
        neighbor = params + np.random.normal(0, 1, size=params.shape)
        return np.clip(neighbor, search_range[0], search_range[1])

    def simulated_annealing(self, search_range, func, alpha=0.9, T_0=1000, T_min=1e-6):
        best_param = None
        best_params = []
        best_value = np.inf
        best_values = []
        all_params = []
        all_values = []

        # Počáteční teplota
        T = T_0

        # Vygenerování počátečního řešení
        current_param = self.generate_initial_solution(search_range)
        current_value = func(current_param)
        
        # Přidání do seznamu všech parametrů a hodnot
        best_param = current_param
        best_value = current_value
        best_params.append(best_param)
        best_values.append(best_value)
        
        # While cyklus pro simulované žíhání
        while T > T_min:
            # Vygenerování souseda
            neighbor_param = self.generate_neighbor(current_param,search_range)
            neighbor_value = func(neighbor_param)

            # zaznamenání všech parametrů a hodnot
            all_params.append(neighbor_param)
            all_values.append(neighbor_value)

            # Přijetí nebo odmítnutí nového řešení
            if neighbor_value < current_value:
                current_param = neighbor_param
                current_value = neighbor_value
            else:
                # Odmítnutí nového řešení s pravděpodobností
                r = np.random.uniform(0, 1)
                # Pokud je pravděpodobnost menší než exp(-deltaE/T), tak se nové řešení přijme
                if r < np.exp(-(neighbor_value - current_value) / T):
                    current_param = neighbor_param
                    current_value = neighbor_value

            # Aktualizace nejlepšího řešení
            if current_value < best_value:
                best_value = current_value
                best_param = current_param
                best_params.append(best_param)
                best_values.append(best_value)

            # Snížení teploty
            T *= alpha

        print(f"Best value: {best_value}\nBest parameters: {best_param}\n")
        return best_params, best_values


    def hill_climbing(self, search_range, step, func, max_iter=1000, tolerance=1e-6):
        params = np.random.uniform(search_range[0], search_range[1], 2)
        best_value = func(params)
        best_params = params
        best_params_list = [params]
        best_values_list = [best_value]
        
        iterations = 0
        while iterations < max_iter:
            neighbors = self.generate_neighbors(params, step)
            
            # "Clipnutí" hodnot do rozsahu definičního oboru
            neighbors = [np.clip(neighbor, search_range[0], search_range[1]) for neighbor in neighbors]
            
            improved = False
            for neighbor in neighbors:
                value = func(neighbor)
                if value < best_value:
                    best_value = value
                    best_params = neighbor
                    best_params_list.append(neighbor)
                    best_values_list.append(value)
                    improved = True
            
            if not improved:
                # Pokud nedošlo k velkému zlepšení, ukončí se vyhledávání
                if abs(best_value - func(params)) < tolerance:
                    break
            
            params = best_params
            iterations += 1
        
        print(f"Best value: {best_value}\nBest parameters: {best_params}\n")
        return best_params_list, best_values_list



    def generate_neighbors(self, params, step):
            neighbors = []
            for i in range(len(params)):
                for delta in [-step, step]:
                    neighbor = np.copy(params)
                    neighbor[i] += delta
                    neighbors.append(neighbor)
            return neighbors

    def plot_neighbors(self, best_params_list, best_values_list, neighbors):
            fig = go.Figure()
            
            best_params_list = np.array(best_params_list)
            best_values_list = np.array(best_values_list)
            fig.add_trace(go.Scatter3d(
                x=best_params_list[:, 0], 
                y=best_params_list[:, 1], 
                z=best_values_list, 
                mode='markers+lines',
                marker=dict(size=5, color='blue'),
                name='Best Points'
            ))
            
            neighbors = np.array(neighbors)
            neighbor_values = [self.sphere(neighbor) for neighbor in neighbors]
            fig.add_trace(go.Scatter3d(
                x=neighbors[:, 0], 
                y=neighbors[:, 1], 
                z=neighbor_values, 
                mode='markers',
                marker=dict(size=5, color='green'),
                name='Neighbors'
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Function Value'
                ),
                title='Neighbors Visualization'
            )
            fig.show()

    # Blind search - náhodné hledání nejlepších parametrů
    def blind_search(self,search_range, iterations, func):
        best_param = None
        best_params = []
        best_value = np.inf
        best_values = []
        all_params = []
        all_values = []
        
        for _ in range(iterations):
            params = np.random.uniform(search_range[0], search_range[1], 2)
            value = func(params)
            all_params.append(params)
            all_values.append(value)
            if value < best_value:
                best_value = value
                best_param = params
                best_params.append(params)
                best_values.append(value)
        
        print(f"Best value: {best_value}\n\nBest parameters: {best_param}\n\n\n")
        # print(f"Best values: {best_values}\n\nBest parameters: {best_params}\n\n\n")
        return best_params, best_values

    def sphere(self,params):
        sum = 0
        for p in params:
            sum += p**2
        return sum
    
    def ackley(self, params, a=20, b=0.2, c=2 * np.pi):
        params = np.array(params)
        d = len(params)
        
        sum1 = np.sum(np.square(params))
        sum2 = np.sum(np.cos(c * params))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
    
        return term1 + term2 + a + np.exp(1)
        
    def rastrigin(self,params):
        sum = 0
        for p in params:
            sum += p**2 - 10*np.cos(2*np.pi*p) + 10
        return sum
    
    def rosenbrock(self,params):
        x = params[0]
        y = params[1]
        return (1-x)**2 + 100*(y-x**2)**2
    
    def griewank(self, params):
        sum_part = 0
        prod_part = 1
        for i in range(len(params)):
            sum_part += params[i]**2
            prod_part *= np.cos(params[i] / np.sqrt(i + 1))
        return 1 + (sum_part / 4000) - prod_part
    
    def schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def levy(self, params):
        params = np.array(params)
        d = len(params)
        
        w = 1 + (params - 1) / 4 #w_i

        term1 = np.sin(np.pi * w[0])**2
        sum_term = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term2 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + sum_term + term2
    
    def michalewicz(self, params):
        sum_part = 0
        d = len(params)
        for i in range(d):
            xi = params[i]
            sum_part += np.sin(xi) * (np.sin(((i + 1) * xi**2) / np.pi))**(2 * 10)
        return -sum_part

    def zakharov(self,params):
        sum1 = 0
        sum2 = 0
        for i in range(len(params)):
            sum1 += params[i]**2
            sum2 += 0.5*(i+1)*params[i]
        return sum1 + sum2**2 + sum2**4
