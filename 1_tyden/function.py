import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
class Function:
    def __init__(self,name):
        self.name = name

    def init_grid(self,precision):
        x = np.linspace(-5, 5,precision)
        y = np.linspace(-5, 5, precision)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        return x, y, z

    def evaluate_grid(self, x, y, z, func):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = func([x[i, j], y[i, j]])
        return z

    def plot_function(self, x, y, z, best_params=None, best_values=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface of the function
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
        
        # Plot red points one by one (if provided)
        if best_params is not None and best_values is not None:
            # print(f"\n\n Length values: {len(best_values)}\n\n Length parameters: {len(best_params)}")

            best_params = np.array(best_params)
            for i in range(len(best_values)):
                ax.scatter(best_params[i, 0], best_params[i, 1], best_values[i], color='red', s=50, label='Search Points' if i == 0 else "")
                plt.pause(1.0)  # Pause to visualize the progress
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'{self.name} Function Value')
        ax.set_title(f'{self.name} Function Plot')
        ax.legend()
        plt.show()



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
        
        # print(f"Best value: {best_value}\n\nBest parameters: {best_param}\n\n\n")
        # print(f"Best values: {best_values}\n\nBest parameters: {best_params}\n\n\n")
        return best_params, best_values, all_params, all_values

    def sphere(self,params):
        sum = 0
        for p in params:
            sum += p**2
        return sum
    
    def ackley(self,params):
        x = params[0]
        y = params[1]
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20
    
    def rastrigin(self,params):
        sum = 0
        for p in params:
            sum += p**2 - 10*np.cos(2*np.pi*p) + 10
        return sum
    
    def rosenbrock(self,params):
        x = params[0]
        y = params[1]
        return (1-x)**2 + 100*(y-x**2)**2
    
    def griewank(self,params):
        sum = 0
        prod = 1
        for i in range(len(params)):
            sum += params[i]**2
            prod *= np.cos(params[i]/np.sqrt(i+1))
        return 1 + sum/4000 - prod
    
    def schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

        


    def levy(self,params):
        x = params[0]
        y = params[1]
        return np.sin(3*np.pi*x)**2 + (x-1)**2*(1 + np.sin(3*np.pi*y)**2) + (y-1)**2*(1 + np.sin(2*np.pi*y)**2)
    
    def michalewicz(self,params):
        sum = 0
        for i in range(len(params)):
            sum += np.sin(params[i])*np.sin((i+1)*params[i]**2/np.pi)**20
        return -sum
    
    def zakharov(self,params):
        sum1 = 0
        sum2 = 0
        for i in range(len(params)):
            sum1 += params[i]**2
            sum2 += 0.5*(i+1)*params[i]
        return sum1 + sum2**2 + sum2**4

