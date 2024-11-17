import numpy as np
class constants:
    PRECISION = 50
    ITERATIONS = 1000
    STEP = 0.2
    DIMENSIONS = 2

    SPHERE_RANGE = [-5.12, 5.12]
    ACKLEY_RANGE = [-30, 30]
    SCHWEFEL_RANGE = [-500, 500]
    RASTRIGIN_RANGE = [-5.12, 5.12]
    ROSENBROCK_RANGE = [-2, 2]
    GRIEWANK_RANGE = [-5, 10] # [-600, 600] [-10,10]
    LEVY_RANGE = [-10, 10]
    MICHALEWICZ_RANGE = [0, np.pi]
    ZAKHAROV_RANGE = [-5, 5]

    FUNCTION_NAMES = {1:"SPHERE" ,2:"ACKLEY" ,3:"RASTRIGIN" ,4:"ROSENBROCK" ,5:"GRIEWANK" ,6:"SCHWEFEL" ,7:"LEVY" ,8:"MICHALEWICZ" ,9:"ZAKHAROV" }
    

    # Particle Swarm

    # PSO Parameters
    BOUNDS = (-10, 10)
    NUM_PARTICLES = 30
    ITERATIONS_PSO = 50
    INERTIA_WEIGHT = 0.5
    COGNITIVE_COEFF = 1.5
    SOCIAL_COEFF = 2.0
