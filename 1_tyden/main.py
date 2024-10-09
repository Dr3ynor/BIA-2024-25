import numpy as np
from function import Function
from solution import Solution
from constants import constants

if __name__ == '__main__':
    while True:
        print("Available functions:\n1. Sphere\n2. Ackley\n3. Rasstrigin\n4. Rosenbrock\n5. Griewank\n6. Schwefel\n7. Levy\n8. Michalewicz\n9. Zakharov\n")
        function_number = input("Select Function:")
        # print(f"ACKLEY RANGE: {constants.ACKLEY_RANGE}")
        try:
            function_number = int(function_number)
            if function_number not in range(1, 10):
                raise ValueError("Invalid function number")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
        function = Function(str(constants.FUNCTION_NAMES[function_number]))
        # Generate a grid of points
        x, y, z = function.init_grid()
        # Evaluate the specified function at each point in the grid
        function_to_evaluate = getattr(function, constants.FUNCTION_NAMES[function_number].lower())
        z = function.evaluate_grid(x, y, z, function_to_evaluate)

        # Plot the results
        # function.plot_function(x, y, z)

        # Perform a blind search for the minimum of the specified function
        search_range = getattr(constants, f"{constants.FUNCTION_NAMES[function_number].upper()}_RANGE")
        print(f"Searched range of {function_to_evaluate}: {search_range}")
        iterations = 1000

        best_params, best_values, all_params, all_values = function.blind_search(search_range, iterations, function_to_evaluate)        
        
        function.plot_function(x, y, z,best_params,best_values)

        if input("Do you want to continue? (y/n): ") == 'n':
            break
    