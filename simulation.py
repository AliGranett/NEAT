import phenotype
import numpy as np
import time
from phenotype import FeedForwardNet, Hyperparameters
from typing import Callable

class Simulation:
    def __init__(self ,inputs: int, outputs: int, population: int, hyperparams: dict, fitness_function: Callable, eval_function: Callable, verbose: bool = False):
        self.inputs = inputs
        self.outputs = outputs
        self.population = population
        self.hyperparams = Hyperparameters(**hyperparams)
        self.fitness_function = fitness_function
        self.eval_function = eval_function
        self.ffn = None  # FeedForward Network instance
        self.verbose = verbose

    def reset_network(self):
        """Initialize the FeedForward Network with the specified hyperparameters."""
        self.ffn = FeedForwardNet(self.inputs, self.outputs, self.population, self.hyperparams)
        self.ffn.generate()

    def run_simulation(self):
        """Run the simulation, evolving the network through generations."""
        if self.verbose:
            print("Training...")
        self.elapsed_generations = 0
        while self.ffn.should_evolve():
            self.ffn.evaluate_parallel(self.eval_function)

            # Print training progress
            current_gen = self.ffn.get_generation()
            current_best = self.ffn.get_fittest()
            if self.verbose:
                print(f"Current Fittness: {current_best.get_fitness() * 100:.2f}% | Generation {current_gen}/{self.hyperparams.max_generations}")
                if current_best.get_fitness() >= self.hyperparams.max_fitness:
                    break
            self.elapsed_generations += 1

    def display_results(self):
        """Evaluate and display the results of the best genome after the simulation."""
        best = self.ffn.get_fittest()
        fit = self.eval_function(best)
        if self.verbose:
            print("\nEvaluation of Best Network:")
            print(f"Overall Fitness: {fit * 100:.2f}%")

    def evaluate_best_network(self, test_inputs, expected_outputs):
        """
        Evaluate the best network with a set of test inputs and compare against expected outputs.
        Parameters:
            test_inputs (list): A list of inputs to test the network with.
            expected_outputs (list): The expected outputs for each input.
        """
        best = self.ffn.get_fittest()
        actual_outputs = [best.forward(input_data)[0] for input_data in test_inputs]

        # Assuming self.fitness_function can compare a list of expected and actual outputs
        fit = self.fitness_function(expected_outputs, actual_outputs)

        print("\nEvaluation of Best Network:")
        for i, input_data in enumerate(test_inputs):
            print(f"Input: {input_data} | Expected Output: {expected_outputs[i]} | Actual Output: {actual_outputs[i]:.3f}")
        print(f"\nOverall Fitness: {fit * 100:.2f}%")

    def display_best_network_structure(self):
        """
        Displays the structure of the best network in the population.
        """
        best = self.ffn.get_fittest()
        edges = best.get_edges()
        nodes = best.get_nodes()

        graph = {}
        for (i, j), edge in edges.items():
            if not edge.enabled:
                continue
            if i not in graph:
                graph[i] = []
            graph[i].append(j)

        num_inputs = self.inputs
        num_outputs = self.outputs
        num_hidden = len(nodes) - (num_inputs + num_outputs)
        num_active_connections = sum(1 for edge in edges.values() if edge.enabled)

        if self.verbose:
            print("\nBest Network Structure:")
            print(f"# of Hidden Nodes: {num_hidden}")
            print(f"# of Active Connections: {num_active_connections}")

            print("\nBest Network Structure:")
            for k in sorted(graph):
                print(f"{k} -> {graph[k]}")

        return num_hidden, num_active_connections, self.elapsed_generations, best._fitness