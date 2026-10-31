import time

import pandas as pd

import xor
from simulation import Simulation


def run_baseline():
    """Run the XOR experiment `xor.runs` times and summarise the results."""
    start = time.time()
    sim = Simulation(
        inputs=2,
        outputs=1,
        population=150,
        hyperparams=xor.hyperparams,
        fitness_function=xor.fitness,
        eval_function=xor.evaluate,
        verbose=True,
    )

    # Per run we record: hidden nodes, active connections, species,
    # generations to solution and best fitness.
    metrics = []
    for i in range(xor.runs):
        sim.reset_network()
        sim.run_simulation()
        sim.display_results()
        results = sim.display_best_network_structure()
        metrics.append(results)

        num_hidden, active_connections, species, generations, best_fitness = results
        print(
            f"Run {i + 1:>3} complete | "
            f"best fitness: {int(best_fitness * 100):>3}%, "
            f"{num_hidden} hidden nodes, "
            f"{active_connections:>3} active connections, "
            f"{species:>3} species, "
            f"{generations:>3} generations to solution | "
            f"{time.time() - start:.2f}s elapsed."
        )

    df = pd.DataFrame(
        metrics,
        columns=[
            "num_hidden",
            "active_connections",
            "species",
            "generations_to_solution",
            "best_fitness",
        ],
    )
    df["best_fitness"] = df["best_fitness"] * 100

    print("\nSummary (all runs):")
    print(df.describe())

    solved = df[df["generations_to_solution"] < xor.hyperparams["max_generations"]]
    print("\nSummary (successful runs):")
    print(solved.describe())

    print(f"\nTotal time: {time.time() - start:.2f}s.")


if __name__ == "__main__":
    run_baseline()
