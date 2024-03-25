import math
from turtle import distance
import activations
from functools import partial

input_size = 2
output_size = 1

population = 150

# The experiment says until we find solution, but if NEAT is stuck, halt early
max_generations = 100
runs = 100

activation_function = partial(activations.sigmoid, alpha=4.9)
fitness_threshold = 3.9 / 4

breed_probs = {
    'sexual' : 0.75,
    'asexual' : 0.25
}

mut_probs = {
    'node' : 0.03,
    'edge' : 0.05,
    'weight_perturb': 0.8*0.9, # shift weight
    'weight_set': 0.8*0.1,
    'bias_perturb': 0.12*0.3, # shift bias
    'bias_set': 0.12*0.7
}

distance_weights = {
    'edge': 1.0,
    'weight': 1.0,
    'bias': 0.4

}

interspecies_mating = 0.001
patience = 15
champion_population_threshold = 5

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