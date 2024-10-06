import numpy as np
from function import Function
from solution import Solution

if __name__ == '__main__':
    function = Function("")

    # Generate a grid of points
    x, y, z = function.init_grid()
    # Evaluate the specified function at each point in the grid
    z = function.evaluate_grid(x, y, z, function.zakharov)

    # Plot the results
    function.plot_function(x, y, z)
