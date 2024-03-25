from simulation import Simulation
import pandas as pd
import random
import time
if __name__ == '__main__':
    # XOR problem
    import xor
    random.seed(0)
    sim_xor = Simulation(
        inputs=xor.input_size,
        outputs=xor.output_size,
        population=xor.population,
        hyperparams={
            "max_generations" : xor.max_generations, 
            "max_fitness" : xor.fitness_threshold,
            "default_activation" : xor.activation_function,
            "breed_probabilities" : xor.breed_probs,
            "mutation_probabilities" : xor.mut_probs,
            "distance_weights" : xor.distance_weights,
            "interspecies_mating_rate" : xor.interspecies_mating,
            "patience" : xor.patience,
            "champion_population_threshold" : xor.champion_population_threshold

        },
        fitness_function=xor.fitness,
        eval_function=xor.evaluate,
        verbose=False
    )

    start = time.time()
    metrics = []
    for i in range(xor.runs):
        sim_xor.setup_network()
        sim_xor.run_simulation()
        sim_xor.display_results()
        results = sim_xor.display_best_network_structure()
        metrics.append(results)
        num_hidden, active_connections, generations_to_solution, best_fitness = results
        print(f"Run {i+1:>3} complete | Best accuracy: {int(best_fitness*100):>3}%, {num_hidden} hidden nodes, {active_connections:>3} active connections, {generations_to_solution:>3} generations to solution | {time.time() - start:.2f} seconds elapsed.")

    df = pd.DataFrame(metrics, columns=["num_hidden", "active_connections", "generations_to_solution", "best_fitness"])
    df['best_fitness'] = df['best_fitness'] * 100
    print("\nSummary (all):")
    print(df.describe())
    print("\nSummary (successful runs):")
    print(df[df['generations_to_solution'] < xor.max_generations].describe())
    end = time.time() - start
    print(f"Time took: {end:,} seconds.")