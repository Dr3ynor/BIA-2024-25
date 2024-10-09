import numpy as np
import matplotlib.pyplot as plt

class Function:
    def __init__(self,name):
        self.name = name

    def init_grid(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        return x,y,z

    def evaluate_grid(self, x, y, z, func):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = func([x[i, j], y[i, j]])
        return z

    def plot_function(self,x,y,z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'{self.name} Function Value')
        ax.set_title(f'{self.name} Function Plot')
        plt.show()

    def blind_search(self,search_range, iterations, func):
        best_params = None
        best_value = np.inf
        for i in range(iterations):
            params = np.random.uniform(search_range[0], search_range[1], 2)
            value = func(params)
            if value < best_value:
                best_value = value
                best_params = params
        return best_params, best_value


    def sphere(self,params):
        sum = 0
        for p in params:
            sum += p**2
        return sum
    
    def ackley(self,params):
        x = params[0]
        y = params[1]
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20
    
    def rasstrigin(self,params):
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

