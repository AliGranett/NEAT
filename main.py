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
            "default_activation" : xor.activation_function
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
        num_hidden, active_connections, generations_to_solution = results
        print(f"Run {i+1} complete | {num_hidden} hidden nodes, {active_connections} active connections, {generations_to_solution} generations to solution | {time.time() - start:.2f} seconds elapsed.")

    num_hidden, active_connections, generations_to_solution = zip(*metrics)
    df = pd.DataFrame(metrics, columns=["num_hidden", "active_connections", "generations_to_solution"])
    print(df.describe())
    end = time.time() - start
    print(f"Time took: {end:,} seconds.")