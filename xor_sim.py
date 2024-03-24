import math
import phenotype

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    
    total_error = sum(abs(e - o) for e, o in zip(expected, output))
    
    # Subtracting the total error from 4
    fitness_value = 4 - total_error
    
    # Squaring the result to disproportionately reward closer solutions
    fitness_score = fitness_value**2
    
    return fitness_score

    # s = 0
    # for i in range(4):
    #     s += (expected[i] - output[i])**2
    # return 1/(1 + math.sqrt(s))

def evaluate(genome):
    """Evaluates the current genome."""
    f1 = genome.forward([0.0, 0.0])[0]
    f2 = genome.forward([1.0, 0.0])[0]
    f3 = genome.forward([0.0, 1.0])[0]
    f4 = genome.forward([1.0, 1.0])[0]
    return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

def main():
    hyperparams = phenotype.Hyperparameters()
    hyperparams.max_generations = 300

    ffn = phenotype.FeedForwardNet(2, 1, 150, hyperparams)
    ffn.generate()
    
    print("Training...")
    while ffn.should_evolve():
        ffn.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = ffn.get_generation()
        current_best = ffn.get_fittest()
        print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
            current_best.get_fitness() * 100, 
            current_gen, 
            hyperparams.max_generations
        ))

    best = ffn.get_fittest()
    f1 = best.forward([0.0, 0.0])[0]
    f2 = best.forward([1.0, 0.0])[0]
    f3 = best.forward([0.0, 1.0])[0]
    f4 = best.forward([1.0, 1.0])[0]
    fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

    edges = best.get_edges()
    graph = {}
    for (i, j) in edges:
        if not edges[(i, j)].enabled:
            continue
        if i not in graph:
            graph[i] = []
        graph[i].append(j)

    print()
    print(f"Best network structure: {best.get_num_nodes()} nodes")
    for k in graph:
        print(f"{k} - {graph[k]}")
    print()
    print("Accuracy: {:.2f}%".format(fit * 100))
    print("0 ^ 0 = {:.3f}".format(f1))
    print("1 ^ 0 = {:.3f}".format(f2))
    print("0 ^ 1 = {:.3f}".format(f3))
    print("1 ^ 1 = {:.3f}".format(f4))

if __name__ == "__main__":
    main()