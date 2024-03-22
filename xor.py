import math

input_size = 2
output_size = 1
population = 150
max_generations = 100

fitness_threshold = 3.9

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1/(1 + math.sqrt(s))

def evaluate(genome):
    """Evaluates the current genome."""
    f1 = genome.forward([0.0, 0.0])[0]
    f2 = genome.forward([1.0, 0.0])[0]
    f3 = genome.forward([0.0, 1.0])[0]
    f4 = genome.forward([1.0, 1.0])[0]
    return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])