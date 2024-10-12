from function import Function
from constants import constants

def get_choice(prompt, choices):
    print(prompt)
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    while True:
        try:
            choice_number = int(input("Select: "))
            if choice_number not in range(1, len(choices) + 1):
                raise ValueError("Invalid choice number")
            return choice_number
        except ValueError as e:
            print(f"Error: {e}")

def main():
    function_choices = [
        "Sphere", "Ackley", "Rastrigin", "Rosenbrock", "Griewank",
        "Schwefel", "Levy", "Michalewicz", "Zakharov"
    ]
    algorithm_choices = ["Blind Search", "Hill Climbing"]

    while True:
        function_number = get_choice("Available functions:", function_choices)
        algorithm_choice = get_choice("Available algorithms:", algorithm_choices)

        function_name = constants.FUNCTION_NAMES[function_number]
        search_range = getattr(constants, f"{function_name.upper()}_RANGE")
        print(f"Searched range: {search_range}")
        
        function = Function(function_name)
        x, y, z = function.init_grid(constants.PRECISION, search_range)
        function_to_evaluate = getattr(function, function_name.lower())
        z = function.evaluate_grid(x, y, z, function_to_evaluate)

        if algorithm_choice == 1:
            best_params, best_values = function.blind_search(search_range, constants.ITERATIONS, function_to_evaluate)
        elif algorithm_choice == 2:
            best_params, best_values = function.hill_climbing(search_range, constants.STEP, function_to_evaluate)
        
        function.plot_function(x, y, z, best_params, best_values)
        
        if input("Do you want to continue? (y/n): ").lower() == 'n':
            break

if __name__ == '__main__':
    main()
