import matplotlib.pyplot as plt
import numpy as np

PLOT_SIZE = 200

def plot_2d_graph(points):
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.title('Travelling Salesman Problem')
    plt.axis([0, PLOT_SIZE, 0, PLOT_SIZE])
    plt.show()


def generate_random_points(number_of_points=10):
    points = np.random.rand(number_of_points, 2) * PLOT_SIZE
    return points


def euclidian_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generic_algorithm(points):
    pass

def main():
    points = generate_random_points()
    plot_2d_graph(points)


if __name__ == "__main__":
    main()
