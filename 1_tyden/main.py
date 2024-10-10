from function import Function
from constants import constants

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

def main():
    while True:
        function_number = get_function_choice()
        function_name = constants.FUNCTION_NAMES[function_number]
        search_range = getattr(constants, f"{function_name.upper()}_RANGE")
        print(f"Searched range:{search_range}")
        function = Function(function_name)
        x, y, z = function.init_grid(constants.PRECISION,search_range)

        function_to_evaluate = getattr(function, function_name.lower())
        z = function.evaluate_grid(x, y, z, function_to_evaluate)

        best_params, best_values = function.blind_search(search_range, constants.ITERATIONS, function_to_evaluate)        
        
        function.plot_function(x, y, z, best_params, best_values)

        if input("Do you want to continue? (y/n): ") == 'n':
            break

if __name__ == '__main__':
    main()