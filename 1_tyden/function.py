import numpy as np
import plotly.graph_objects as go

class Function:
    def __init__(self,name):
        self.name = name
        print(f"Function: {self.name}")

    def get_function_choice():
        print("Available functions:\n1. Sphere\n2. Ackley\n3. Rastrigin\n4. Rosenbrock\n5. Griewank\n6. Schwefel\n7. Levy\n8. Michalewicz\n9. Zakharov\n")
        while True:
            try:
                function_number = int(input("Select Function:"))
                if function_number not in range(1, 10):
                    raise ValueError("Invalid function number")
                return function_number
            except ValueError as e:
                print(f"Error: {e}")

    def init_grid(self,precision,range):
        x = np.linspace(range[0], range[1],precision)
        y = np.linspace(range[0], range[1], precision)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        return x, y, z

    def evaluate_grid(self, x, y, z, func):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = func([x[i, j], y[i, j]])
        return z

    def plot_function(self, x, y, z, best_params=None, best_values=None):
        fig = go.Figure()
        # Plot the surface of the function
        fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale='Viridis', opacity=0.8))
        
        # Plot red points
        if best_params is not None and best_values is not None:
            best_params = np.array(best_params)
            
            # Plot the search points as red dots at the correct Z value
            for i in range(len(best_values)):
                z_val = best_values[i]
                if i == len(best_values)-1:
                    fig.add_trace(go.Scatter3d(
                        x=[best_params[i, 0]], 
                        y=[best_params[i, 1]], 
                        z=[z_val], 
                        mode='markers',
                        marker=dict(size=7, color='yellow'),
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

    def blind_search(self,search_range, iterations, func):
        best_param = None
        best_params = []
        best_value = np.inf
        best_values = []
        all_params = []
        all_values = []
        
        for i in range(iterations):
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
        
        # Compute w_i values
        w = 1 + (params - 1) / 4  # Using the formula for w_i

        # Calculate the Levy function
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
