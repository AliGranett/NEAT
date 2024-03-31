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


if __name__ == '__main__':
    # XOR problem
    random.seed(0)
    sim_xor = Simulation(
        inputs=2,
        outputs=1,
        population=250,
        hyperparams=xor.hyperparams,
        fitness_function=xor.fitness,
        eval_function=xor.evaluate,
        verbose=True
    )
    start = time.time()
    # store results :{num_hidden, active_connections, generations_to_solution, best_fitness}
    metrics = []

    for i in range(xor.runs):
        sim_xor.reset_network()
        sim_xor.run_simulation()
        sim_xor.display_results()
        results = sim_xor.display_best_network_structure()
        metrics.append(results)
        num_hidden, active_connections, num_inactive_connections ,generations_to_solution, best_fitness = results
        print(f"Run {i+1:>3} complete | Best accuracy: {int(best_fitness*100):>3}%, {num_hidden} hidden nodes, {active_connections:>3} active connections (inactive: {num_inactive_connections:>3}) ,{generations_to_solution:>3} generations to solution | {time.time() - start:.2f} seconds elapsed.")

    df = pd.DataFrame(metrics, columns=[
                      "num_hidden", "active_connections", "generations_to_solution", "best_fitness"])
    df['best_fitness'] = df['best_fitness'] * 100
    print("\nSummary (all):")
    print(df.describe())
    print("\nSummary (successful runs):")
    print(df[df['generations_to_solution'] < xor.hyperparams['max_generations']].describe())
    end = time.time() - start
    print(f"Time took: {end:,} seconds.")
