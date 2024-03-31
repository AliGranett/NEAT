from simulation import Simulation
import pandas as pd
import random
import time
import xor

"""
TODO LIST:
1. add inter-species mating
2. DONE add champion cloning for species with more than (5 : check paper) members
3. DONE add patience (max history) for species with no improvement in fitness
4. add enable/disable edge mutation
"""

experiment = 'baseline'

if __name__ == '__main__':
    start = time.time()
    if experiment == 'baseline':
        sim_xor = Simulation(
            inputs=2,
            outputs=1,
            population=150,
            hyperparams=xor.hyperparams,
            fitness_function=xor.fitness,
            eval_function=xor.evaluate,
            verbose=True
        )
        # store results :{num_hidden, active_connections, #species ,generations_to_solution, best_fitness}
        metrics = []

        for i in range(xor.runs):
            sim_xor.reset_network()
            sim_xor.run_simulation()
            sim_xor.display_results()
            results = sim_xor.display_best_network_structure()
            metrics.append(results)
            num_hidden, active_connections, species ,generations_to_solution, best_fitness = results
            print(f"Run {i+1:>3} complete | Best accuracy: {int(best_fitness*100):>3}%, {num_hidden} hidden nodes, {active_connections:>3} active connections, species: {species:>3}) ,{generations_to_solution:>3} generations to solution | {time.time() - start:.2f} seconds elapsed.")

        df = pd.DataFrame(metrics, columns=[
                        "num_hidden", "active_connections", "species" ,"generations_to_solution", "best_fitness"])
        df['best_fitness'] = df['best_fitness'] * 100
        print("\nSummary (all):")
        print(df.describe())
        print("\nSummary (successful runs):")
        print(df[df['generations_to_solution'] < xor.hyperparams['max_generations']].describe())
        end = time.time() - start
        print(f"Time took: {end:,} seconds.")
    else:
        max_gens = 300
        exp_hyperparams = xor.hyperparams
        exp_hyperparams['max_generations'] = max_gens
        experiments = ['a', 'delta']
        a_lims = [0.5, 0.7, 1, 2]
        deltas = [0.5, 0.75, 1, 2]
        if experiment == 'a':
            for a_lim in a_lims:
                exp_hyperparams['distributions_parameters']['uniform_range'] = a_lim
                a_sim = Simulation(
                            inputs=2,
                            outputs=1,
                            population=150,
                            hyperparams=exp_hyperparams,
                            fitness_function=xor.fitness,
                            eval_function=xor.evaluate,
                            verbose=False
                        )
                a_sim.reset_network()
                a_sim.run_simulation()
                a_sim.display_results()
                results = a_sim.display_best_network_structure()
                num_hidden, active_connections, species ,generations_to_solution, best_fitness = results
                print(f"Run for {a_lim=} complete | Best accuracy: {int(best_fitness*100):>3}%, {num_hidden} hidden nodes, {active_connections:>3} active connections, species: {species:>3}) ,{generations_to_solution:>3} generations to solution | {time.time() - start:.2f} seconds elapsed.")

                