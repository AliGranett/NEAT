from simulation import Simulation
import time
if __name__ == '__main__':
    # XOR problem
    import xor
    sim_xor = Simulation(
        inputs=xor.input_size,
        outputs=xor.output_size,
        population=xor.population,
        hyperparams={
            "max_generations" : xor.max_generations, 
            "max_fitness" : xor.fitness_threshold,
        },
        fitness_function=xor.fitness,
        eval_function=xor.evaluate
    )

    sim_xor.setup_network()
    start = time.time()
    sim_xor.run_simulation()
    sim_xor.display_results()
    end = start - time.time()

    print(f"Time took: {end - start:.2} seconds.")