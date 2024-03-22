import phenotype
import numpy as np
import time
from phenotype import FeedForwardNet, Hyperparameters
from typing import Callable
from functools import partial
# from tqdm import tqdm

class Simulation:
    def __init__(self ,inputs: int, outputs: int, population: int, hyperparams: dict, fitness_function: Callable, eval_function: Callable):
        self.inputs = inputs
        self.outputs = outputs
        self.population = population
        self.hyperparams = Hyperparameters(**hyperparams)
        self.fitness_function = fitness_function
        self.eval_function = eval_function
        self.ffn = None  # FeedForward Network instance

    def setup_network(self):
        """Initialize the FeedForward Network with the specified hyperparameters."""
        self.ffn = phenotype.FeedForwardNet(self.inputs, self.outputs, self.population, self.hyperparams)
        self.ffn.generate()

    def run_simulation(self):
        """Run the simulation, evolving the network through generations."""
        print("Training...")
        while self.ffn.should_evolve():
            self.ffn.evaluate_parallel(self.wrapper_eval_function)

            # Print training progress
            current_gen = self.ffn.get_generation()
            current_best = self.ffn.get_fittest()
            print(f"Current Accuracy: {current_best.get_fitness() * 100:.2f}% | Generation {current_gen}/{self.hyperparams.max_generations}")

    def wrapper_eval_function(self, genome):
        """A wrapper for the evaluation function that applies the fitness function."""
        return self.eval_function(genome)

    def display_results(self):
        """Evaluate and display the results of the best genome after the simulation."""
        best = self.ffn.get_fittest()
        fit = self.eval_function(best)
        print(f"\nFinal Accuracy: {fit * 100:.2f}%")