import math
import activations

input_size = 2
output_size = 1
population = 150
max_generations = float('inf')
fitness_threshold = 3.9 / 4
activation_function = activations.sigmoid
runs = 5

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    s = sum(abs(e - o) for e, o in zip(expected, output))
    dist = 4 - s
    dist = dist**2
    return dist

def evaluate(genome):
    """Evaluates the current genome."""
    f1 = genome.forward([0.0, 0.0])[0]
    f2 = genome.forward([1.0, 0.0])[0]
    f3 = genome.forward([0.0, 1.0])[0]
    f4 = genome.forward([1.0, 1.0])[0]
    return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])